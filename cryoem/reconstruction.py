import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import imread, imwrite
# from cryoem.projections import RotationMatrix
import astra


def RotationMatrix(angles, transposed=False):
    vectors = np.zeros((angles.shape[0],12))
    vectors[:,0:3] = [0, 0, 1]

    # center of detector
    vectors[:,3:6] = 0
 
    # vector from detector pixel (0,0) to (0,1)
    vectors[:,6:9] = [1, 0, 0]
     
    # vector from detector pixel (0,0) to (1,0)
    vectors[:,9:12]  = [0, 1, 0]
 
    # create rotation matrix
    c1 = np.cos(angles[:,0]).reshape(-1,1,1)
    c2 = np.cos(angles[:,1]).reshape(-1,1,1)
    c3 = np.cos(angles[:,2]).reshape(-1,1,1)
                        
    s1 = np.sin(angles[:,0]).reshape(-1,1,1)
    s2 = np.sin(angles[:,1]).reshape(-1,1,1)
    s3 = np.sin(angles[:,2]).reshape(-1,1,1)
    vector = vectors[0,:]

    R = np.concatenate([np.concatenate([c1*c2*c3-s1*s3, c1*s3+c2*c3*s1 , -c3*s2],axis=2),\
                        np.concatenate([-c3*s1-c1*c2*s3,    c1*c3-c2*s1*s3 ,   s2*s3],axis=2),\
                        np.concatenate( [c1*s2,             s1*s2          ,   c2],axis=2)],axis=1)
    if transposed:
        R = np.transpose(R, (0, 2, 1))
    
    # rotate previous values
    vectors[:,0:3] = np.matmul(R,vector[0:3])
    vectors[:,6:9] = np.matmul(R,vector[6:9])
    vectors[:,9:12] = np.matmul(R,vector[9:12])

    return vectors

def reconstruct(projections, angles, mrc_filename=True, transposed=False):
    # Generate orientation vectors based on angles
    orientation_vectors   = RotationMatrix(angles, transposed)

    # Reshape projections correctly 
    projections1 = np.transpose(projections, (1, 0, 2))
    
    # Get projection dimension
    proj_size = projections1.shape[0]

    # Create projection 2D geometry in ASTRA
    proj_geom = astra.create_proj_geom('parallel3d_vec', proj_size, proj_size, orientation_vectors)
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

    # Cleanup.
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(reconstruction_id)
    astra.data3d.delete(projections_id)

    # Save reconstruction to mrc file for chimera
    if mrc_filename:
        with mrcfile.new(mrc_filename) as mrc:
            mrc.set_data(reconstruction)
        
    return reconstruction

def reconstruct_from_file(input_file, limit=3000, mrc_filename=None):
    data = np.load(f'data/{input_file}.npz')
    projections, angles = data["arr_0"].astype(np.float64)[:limit, :, :], data["arr_1"].astype(np.float64)[:limit, :]

    return reconstruct(projections, angles, mrc_filename)