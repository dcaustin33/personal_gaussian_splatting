from torch import nn
import torch
from typing import Tuple
from tqdm import tqdm

from splat.utils import project_points, getWorld2View, extract_gaussian_weight
from splat.read_colmap import qvec2rotmat, qvec2rotmat_matrix


class GaussianScene(nn.Module):
    def __init__(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        e_opacity: float = 0.005,
        divide_scale: float = 1.6,
        gradient_pos_threshold: float = 0.0002,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # nx3 matrix
        self.points = points.clone().requires_grad_(True).to(self.device).float()
        # nx3 matrix
        self.colors = colors.clone().requires_grad_(True).to(self.device).float()
        # nx1 matrix
        self.opacity = torch.tensor(
            [1] * len(points),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        # nx4 matrix
        self.quaternions = torch.tensor(
            [[1, 1, 1, 1]] * len(points),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        # nx3 matrix
        self.scales = torch.ones((points.shape[0], 3), device=self.device)
        with torch.no_grad():
            self.initialize_scale()
        # used for opacity and stopping of pixel
        self.opacity_activation = nn.functional.sigmoid
        self.opacity_threshold = 0.99

        # used for densifying and removing gaussians
        self.e_opacity = e_opacity
        self.divide_scale = divide_scale
        # this corresponds to densify_grad_threshold in the original code
        self.gradient_pos_threshold = gradient_pos_threshold
        self.size_threshold = 20
        self.percent_dense = 0.01  # not sure what this is really
        
    def initialize_scale(self) -> None:
        """Initializes the scale of the covariance from the distance to the three closest points"""
        all_scale = []
        for i in range(self.points.shape[0]):
            point = self.points[i]
            distances = torch.linalg.norm(self.points - point, dim=1)
            distances = distances[distances != 0]
            distances = torch.sort(distances)[0]
            all_scale.append(distances[:3].mean())
        self.scales *= torch.tensor(all_scale).unsqueeze(1)

    # def initialize_scale(self) -> None:
    #     # Compute pairwise distances matrix
    #     point_diffs = self.points.unsqueeze(0) - self.points.unsqueeze(1)
    #     distances = torch.linalg.norm(point_diffs, dim=2)

    #     # Set diagonal to a large number to ignore zero distance to itself
    #     distances.fill_diagonal_(float("inf"))

    #     # Sort distances and take the mean of the three smallest nonzero distances for each point
    #     closest_distances = distances.sort(dim=1).values[:, :3]
    #     all_scale = closest_distances.mean(dim=1)

    #     # Update scales
    #     self.scales *= all_scale.unsqueeze(1)


    def get_3d_covariance_matrix(self) -> torch.Tensor:
        """
        Get the 3D covariance matrix from the scale and rotation matrix
        """
        # noramlize the quaternions
        self.quaternions = nn.functional.normalize(self.quaternions, p=2, dim=1)
        # nx3x3 matrix
        rotation_matrices = torch.stack([qvec2rotmat(q) for q in self.quaternions])
        # nx3x3 matrix
        scale_matrices = torch.stack([torch.diag(s) for s in self.scales])
        scale_rotation_matrix = rotation_matrices @ scale_matrices
        covariance = scale_rotation_matrix @ scale_rotation_matrix.transpose(1, 2)
        return covariance

    def remove_gaussian(
        self,
    ) -> None:
        """Removes the gaussians that are essentially transparent"""
        with torch.no_grad():
            opacities = torch.sigmoid(self.opacity)
            truth = opacities > self.e_opacity
            self.points = self.points[truth]
            self.colors = self.colors[truth]
            self.opacity = self.opacity[truth]
            self.quaternions = self.quaternions[truth]
            self.scales = self.scales[truth]

    def get_points_and_covariance(
        self,
        extrinsic_matrix: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a set of points, we project to 2d and find their 2d covariance matrices

        Args:
            covariance_3d: Nx3x3 tensor of covariance matrices
            extrinsic_matrix: 4x4 tensor translates the points to camera coordinates but still in 3d
            intrinsic_matrix: 3x4 tensor that projects the points to 2d
        """
        points = torch.cat(
            [
                self.points,
                torch.ones(self.points.shape[0], 1, device=self.points.device),
            ],
            dim=1,
        )
        # results in a 4xN tensor
        points_in_camera_coords = torch.matmul(extrinsic_matrix, points.t()).T  # Nx4
        # do not need to divide by 1 as this is always 1
        final_points_in_camera_coords = points_in_camera_coords[
            :, :3
        ] / points_in_camera_coords[:, 3].unsqueeze(1)
        # now we project to 2d

        z_component = points_in_camera_coords[:, 2].unsqueeze(1)
        projected_points, _ = project_points(
            intrinsic_matrix, final_points_in_camera_coords
        )
        # now we find the covariance matrices in 2d
        projected_covariance = []

        f_x = intrinsic_matrix[0, 0]
        f_y = intrinsic_matrix[1, 1]

        # this makes it stored in column major order - something the original code does
        _W = getWorld2View(extrinsic_matrix[:3, :3], extrinsic_matrix[:3, 3]).transpose(
            0, 1
        )
        W = torch.Tensor(
            [
                [_W[0, 0], _W[1, 0], _W[2, 0]],
                [_W[0, 1], _W[1, 1], _W[2, 1]],
                [_W[0, 2], _W[1, 2], _W[2, 2]],
            ]
        )

        covariance_3d = self.get_3d_covariance_matrix()

        for i in range(covariance_3d.shape[0]):
            covariance = covariance_3d[i]
            camera_coords_x = final_points_in_camera_coords[i, 0]
            camera_coords_y = final_points_in_camera_coords[i, 1]
            camera_coords_z = final_points_in_camera_coords[i, 2]
            jacobian = torch.zeros((3, 3), device=points.device)
            jacobian[0, 0] = f_x / camera_coords_z
            jacobian[1, 1] = f_y / camera_coords_z
            jacobian[0, 2] = -f_x * camera_coords_x / (camera_coords_z**2)
            jacobian[1, 2] = -f_y * camera_coords_y / (camera_coords_z**2)
            # import pdb; pdb.set_trace()
            # TODO optimize to do batch mat mul at the end
            T = torch.matmul(jacobian, W.T)
            final_variance = torch.matmul(torch.matmul(T, covariance), T.T)
            projected_covariance.append(final_variance[:2, :2])
        return projected_points, z_component, torch.stack(projected_covariance)

    def covariance_3d_to_2d(
        self,
        points_in_camera_coords: torch.Tensor,
        covariance_3d: torch.Tensor,
        extrinsic_matrix: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given a set of points that we have projected to 2d we now find the 2d covariance matrices

        Args:
            covariance_3d: Nx3x3 tensor of covariance matrices
            points_in_camera_coords: Nx3 tensor of points in camera coordinates, as we retain the z component
            extrinsic_matrix: 4x4 tensor translates the points to camera coordinates but still in 3d
            intrinsic_matrix: 3x4 tensor that projects the points to 2d
        """
        points = torch.cat(
            [
                self.points,
                torch.ones(self.points.shape[0], 1, device=self.points.device),
            ],
            dim=1,
        )
        # results in a 4xN tensor
        points_in_camera_coords = torch.matmul(extrinsic_matrix, points.t()).T

    def get_intersected_tiles(
        self,
        projected_points: torch.Tensor,
        projected_covariance: torch.Tensor,
        height: int,
        width: int,
        tile_size: int = 16,
    ):
        """Returns the intersected tiles for each point. Can be optimized later"""
        eigenvalues = torch.linalg.eigvals(projected_covariance).real
        # get the radius
        radii = torch.sqrt(eigenvalues[:, 0])
        min_y = projected_points[:, 1] - radii
        max_y = projected_points[:, 1] + radii
        min_x = projected_points[:, 0] - radii
        max_x = projected_points[:, 0] + radii
        # now that we have the bounding box, we can find the intersected tiles
        n = projected_points.shape[0]
        tile_matrix = torch.zeros(
            (n, width // 16, height // 16), device=projected_points.device
        )

        # TODO: this could be optimized
        for idx in range(n):
            for x in range(int(min_x[idx].item()), int(max_x[idx].item()), tile_size):
                for y in range(int(min_y[idx].item()), int(max_y[idx].item()), tile_size):
                    x_index = x // tile_size
                    y_index = y // tile_size
                    if x_index >= width // tile_size or x_index < 0:
                        continue
                    if y_index >= height // tile_size or y_index < 0:
                        continue
                    tile_matrix[idx, x // tile_size, y // tile_size] = 1

        return tile_matrix

    # def get_intersected_tiles(
    #     self,
    #     projected_points: torch.Tensor,
    #     projected_covariance: torch.Tensor,
    #     height: int,
    #     width: int,
    #     tile_size: int = 16,
    # ):
    #     """Returns the intersected tiles for each point. Can be optimized later"""
    #     eigenvalues = torch.linalg.eigvals(projected_covariance).real
    #     # get the radius
    #     radii = torch.sqrt(eigenvalues[:, 0])
    #     min_y = (projected_points[:, 1] - radii) / tile_size
    #     max_y = (projected_points[:, 1] + radii) / tile_size
    #     min_x = (projected_points[:, 0] - radii) / tile_size
    #     max_x = (projected_points[:, 0] + radii) / tile_size

    #     # remove points that are outside the image
    #     x_never_in_image = (max_x < 0) | (min_x > width)
    #     y_never_in_image = (max_y < 0) | (min_y > height)
    #     never_in_image = x_never_in_image | y_never_in_image
    #     # remove those points
    #     min_x = min_x[~never_in_image]
    #     max_x = max_x[~never_in_image]
    #     min_y = min_y[~never_in_image]
    #     max_y = max_y[~never_in_image]
    #     projected_points = projected_points[~never_in_image]
    #     projected_covariance = projected_covariance[~never_in_image]
    #     import pdb; pdb.set_trace()

    #     # have indicators for the points that are in the image
    #     n = projected_points.shape[0]
    #     tile_matrix = torch.zeros(
    #         (n, width // 16, height // 16), device=projected_points.device
    #     )

    #     for idx in range(min_x.shape[0]):
    #         x_min = int(min_x[idx].item())
    #         x_max = int(max_x[idx].item())
    #         y_min = int(min_y[idx].item())
    #         y_max = int(max_y[idx].item())
    #         tile_matrix[idx, x_min:x_max, y_min:y_max] = 1

    #     return tile_matrix, never_in_image

    def render_pixel(
        self,
        pixel: torch.Tensor,
        points: torch.Tensor,
        covariance: torch.Tensor,
        colors: torch.Tensor,
        opacity: torch.Tensor,
    ) -> torch.Tensor:
        """Will return a 3x1 color tensor"""
        current_pixel_weight = 0
        pixel_color = torch.zeros(3, device=points.device)
        for point_idx in range(points.shape[0]):
            mean = points[point_idx]
            point_covariance = covariance[point_idx]
            color = colors[point_idx]
            weight = opacity[point_idx] * extract_gaussian_weight(
                pixel, mean, point_covariance
            )
            weight = min(weight, torch.Tensor([0.99]))
            current_pixel_weight += weight
            if current_pixel_weight > self.opacity_threshold:
                break
            pixel_color += weight.view(-1) * color.view(-1)
        return pixel_color

    def render_tile(
        self,
        x: int,
        y: int,
        tile_matrix: torch.Tensor,
        points: torch.Tensor,
        covariance: torch.Tensor,
        z_component: torch.Tensor,
        colors: torch.Tensor,
        opacity: torch.Tensor,
        tile_size: int = 16,
    ):
        upper_left_pixel = torch.Tensor([x * tile_size, y * tile_size])
        in_tile_truth = tile_matrix[:, x, y] == 1
        points_in_tile = points[in_tile_truth]
        covariance_in_tile = covariance[in_tile_truth]
        colors_in_tile = colors[in_tile_truth]
        z_component_in_tile = z_component[in_tile_truth]
        opacity_in_tile = self.opacity_activation(opacity[in_tile_truth])

        # sort by the z component
        sorted_indices = torch.argsort(z_component_in_tile)
        points_in_tile = points_in_tile[sorted_indices]
        covariance_in_tile = covariance_in_tile[sorted_indices]
        colors_in_tile = colors_in_tile[sorted_indices]
        opacity_in_tile = opacity_in_tile[sorted_indices]

        # now we render the tile
        tile = torch.zeros((tile_size, tile_size, 3), device=points.device)
        for i in range(tile_size):
            for j in range(tile_size):
                pixel = torch.Tensor([i, j]) + upper_left_pixel
                tile[i, j] = self.render_pixel(
                    pixel,
                    points_in_tile,
                    covariance_in_tile,
                    colors_in_tile,
                    opacity_in_tile,
                )
        return tile

    def render_scene(
        self,
        projected_points: torch.Tensor,
        projected_covariances: torch.Tensor,
        z_component: torch.Tensor,
        height: int,
        width: int,
        tile_size: int = 16,
    ) -> torch.Tensor:
        """Renders the scene given the projected points and covariance matrices"""
        tile_matrix = (
            self.get_intersected_tiles(
                projected_points=projected_points,
                projected_covariance=projected_covariances,
                height=height,
                width=width,
                tile_size=tile_size,
            )
        )

        scene = torch.zeros(
            (width + tile_size, height + tile_size, 3), device=projected_points.device
        )
        print(
            scene.shape,
            width // tile_size,
            height // tile_size,
        )
        for x in tqdm(range(width // tile_size)):
            for y in range(height // tile_size):
                scene[
                    x * tile_size : x * tile_size + tile_size,
                    y * tile_size : y * tile_size + tile_size,
                ] = self.render_tile(
                    x=x,
                    y=y,
                    tile_matrix=tile_matrix,
                    points=projected_points,
                    covariance=projected_covariances,
                    z_component=z_component,
                    colors=self.colors,
                    opacity=self.opacity,
                    tile_size=tile_size,
                )
        return scene

    def forward(
        self,
        extrinsic_matrix: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
        height: int,
        width: int,
    ):
        projected_points, z_component, projected_covariance = (
            self.get_points_and_covariance(extrinsic_matrix, intrinsic_matrix)
        )
        intersected_tiles = self.get_intersected_tiles(
            projected_points, projected_covariance
        )

        return projected_points, projected_covariance