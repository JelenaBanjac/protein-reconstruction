import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import imread, imwrite
from projections import RotationMatrix
import astra

def run():
    data = np.load('../data/5a1a_projections_and_angles.npz')
    projections, angles = data["arr_0"].astype(np.float64)[:100, :, :], data["arr_1"].astype(np.float64)[:100, :]

    # Generate orientation vectors based on angles
    orientation_vectors   = RotationMatrix(angles)

    # Reshape projections correctly 
    projections1 = np.transpose(projections, (1, 0, 2))
    # Get projection dimension
    proj_size = projections1.shape[0]

    # Create projection 2D geometry in ASTRA
    proj_geom = astra.create_proj_geom('parallel3d_vec', proj_size, proj_size, orientation_vectors)
    print(proj_geom["Vectors"].shape)
    projections_id = astra.data3d.create('-sino', proj_geom, projections1)

    # Create reconstruction.
    vol_geom = astra.creators.create_vol_geom(proj_size, proj_size, proj_size)
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
    alg_cfg = astra.astra_dict('BP3D_CUDA')
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data3d.get(reconstruction_id)


    # Limit and scale reconstruction.
    reconstruction[reconstruction < 0] = 0
    reconstruction /= np.max(reconstruction)
    reconstruction = np.round(reconstruction * 255).astype(np.uint8)
    
    # Save reconstruction.
    if not isdir(output_dir):
        mkdir(output_dir)
    for i in range(detector_rows):
        im = reconstruction[i, :, :]
        im = np.flipud(im)
        imwrite(join(output_dir, 'reco%04d.png' % i), im)

    # Cleanup.
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(reconstruction_id)
    astra.data3d.delete(projections_id)

if __name__=="__main__":
    run()