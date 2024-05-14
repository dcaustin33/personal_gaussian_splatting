import math
import os
from typing import Dict, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from splat.read_colmap import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
)
from splat.schema import BasicPointCloud


def get_intrinsic_matrix(
    f_x: float, f_y: float, c_x: float, c_y: float
) -> torch.Tensor:
    """
    Get the homogenous intrinsic matrix for the camera

    Args:
        f_x: focal length in x
        f_y: focal length in y
        c_x: principal point in x
        c_y: principal point in y
    """
    return torch.Tensor(
        [
            [f_x, 0, c_x, 0],
            [0, f_y, c_y, 0],
            [0, 0, 1, 0],
        ]
    )


def get_extrinsic_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Get the homogenous extrinsic matrix for the camera

    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def project_points(
    projection_matrix: torch.Tensor, points: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projection matrix is the intrinsic matrix a 3x4 matrix, points is a Nx3 matrix.
    Returns a Nx2 matrix
    """
    points = torch.cat(
        [points, torch.ones(points.shape[0], 1, device=points.device)], dim=1
    )
    projected_points = torch.matmul(projection_matrix, points.t()).t()
    z_component = projected_points[:, 2].unsqueeze(1)
    projected_points = projected_points[:, :2] / projected_points[:, 2].unsqueeze(1)

    return projected_points, z_component


def getWorld2View(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    We should translate the points to camera coordinates,
    however we will never use the 4th dimension it seems like so leaving for now
    """
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def extract_gaussian_weight(
    pixel: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    """
    Use the covariance matrix to extract the weight of the point

    Args:
        mean: 1x2 tensor
        covariance: 2x2 tensor
    """
    diff = pixel - mean
    inv_covariance = torch.inverse(covariance)
    return torch.exp(-0.5 * torch.matmul(diff, torch.matmul(inv_covariance, diff.t())))


def fetchPly(path: str):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def build_rotation(r: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def focal2fov(focal: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Rt = torch.zeros((4, 4))

    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt.float()

def getWorld2View2(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.t()  # Use .t() for transpose in PyTorch
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.inverse(Rt)  # Use torch.inverse for matrix inversion in PyTorch
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt.float()  


def getProjectionMatrix(
    znear: torch.Tensor, zfar: torch.Tensor, fovX: torch.Tensor, fovY: torch.Tensor
) -> torch.Tensor:
    """
    This is from the original repo.
    It uses the view to adjust the coordinates to the actual pixel dimensions
    It still retains the z componenet.
    Not 100% sure the difference between this and the intrinsic matrix
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def read_camera_file(colmap_path: str) -> Dict:
    binary_path = os.path.join(colmap_path, "cameras.bin")
    text_path = os.path.join(colmap_path, "cameras.txt")
    if os.path.exists(binary_path):
        cameras = read_cameras_binary(binary_path)
    elif os.path.exists(text_path):
        cameras = read_cameras_text(text_path)
    else:
        raise ValueError
    return cameras


def read_image_file(colmap_path: str) -> Dict:
    binary_path = os.path.join(colmap_path, "images.bin")
    text_path = os.path.join(colmap_path, "images.txt")
    if os.path.exists(binary_path):
        images = read_images_binary(binary_path)
    elif os.path.exists(text_path):
        images = read_images_text(text_path)
    else:
        raise ValueError
    return images


def in_view_frustum(
    points: torch.Tensor, minimum_z: float, view_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Given a view matrix (transforming from world to camera coords) and a minimum
    z value, return a boolean tensor indicating whether the points aree in view.

    points is a Nx3 tensor and we return a N tensor indicating whether the point 
    is in view.
    """
    projected_points = torch.matmul(view_matrix, points.t()).t()
    z_component = projected_points[:, 2]
    return z_component > minimum_z

def ndc2Pix(points: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Convert points from NDC to pixel coordinates
    """
    return (points + 1) * (dimension - 1) * 0.5 

