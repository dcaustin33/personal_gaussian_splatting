from plyfile import PlyData
import numpy as np



if __name__ == "__main__":
    max_sh_degree = 3

    plydata = PlyData.read('/Users/derek/Desktop/personal_gaussian_splatting/point_clouds/gs_Downstairs.ply')
    x = np.asarray(plydata.elements[0]['x'])
    y = np.asarray(plydata.elements[0]['y'])
    z = np.asarray(plydata.elements[0]['z'])
    xyz = np.stack((x, y, z), axis=-1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])



    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)


    import numpy as np
    from scipy.special import sph_harm
    import tqdm

    # Assuming shs is your numpy array of shape (647919, 48)
    # Generate spherical coordinates for the sample directions
    num_samples = 100
    phi = np.linspace(0, 2 * np.pi, num_samples)
    theta = np.linspace(0, np.pi, num_samples)
    phi, theta = np.meshgrid(phi, theta)

    # Flatten the coordinates for easier processing
    phi_flat = phi.flatten()
    theta_flat = theta.flatten()

    # Function to calculate SH basis up to order 7
    def compute_sh_basis(l, m, theta, phi):
        return sph_harm(m, l, phi, theta).real

    # Evaluate all SH bases up to order 7 for all sample directions
    num_sh_coeffs = 48
    order = int(np.sqrt(num_sh_coeffs / 3) - 1)
    sh_basis_functions = np.zeros((num_sh_coeffs // 3, phi_flat.size))

    index = 0
    for l in range(order + 1):  # from 0 to 7
        for m in range(-l, l + 1):
            sh_basis_functions[index, :] = compute_sh_basis(l, m, theta_flat, phi_flat)
            index += 1

    # Compute lighting for each viewpoint
    # Create an array to hold the RGB values for each viewpoint
    rgb_values = np.zeros((shs.shape[0], 3))

    for i in tqdm.tqdm(range(shs.shape[0])):
        sh_coefficients = shs[i, :].reshape(3, -1)
        lighting_r = sh_coefficients[0, :] @ sh_basis_functions
        lighting_g = sh_coefficients[1, :] @ sh_basis_functions
        lighting_b = sh_coefficients[2, :] @ sh_basis_functions

        # Aggregate the lighting values to get a single RGB value
        rgb_values[i, 0] = np.mean(lighting_r)
        rgb_values[i, 1] = np.mean(lighting_g)
        rgb_values[i, 2] = np.mean(lighting_b)
    # remove the nan
    rgb_values = np.nan_to_num(rgb_values)
    
    # Normalize RGB values to range [0, 1]
    rgb_values = (rgb_values - rgb_values.min(axis=0)) / (rgb_values.max(axis=0) - rgb_values.min(axis=0))

    # Display the result for the first viewpoint as an example
    print("RGB values for the first viewpoint:", rgb_values[0])

        
    # save the rgb_values
    np.save("/Users/derek/Desktop/personal_gaussian_splatting/point_clouds/luma/rgb_values.npy", rgb_values)
    np.save("/Users/derek/Desktop/personal_gaussian_splatting/point_clouds/luma/xyz.npy", xyz)
    np.save("/Users/derek/Desktop/personal_gaussian_splatting/point_clouds/luma/scales.npy", scales)
    np.save("/Users/derek/Desktop/personal_gaussian_splatting/point_clouds/luma/rots.npy", rots)
    np.save("/Users/derek/Desktop/personal_gaussian_splatting/point_clouds/luma/opacities.npy", opacities)
