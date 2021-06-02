import numpy as np
from cryoem.rotation_matrices import RotationMatrix
import astra
import mrcfile
from pathlib import Path


def reconstruct(projections, angles, alg_iterations=100, mrc_filename=None, initial_mrc_filename=None ,overwrite=False, vol_shape=None):
    """Method used for protein reconstruction using ASTRA toolbox"""
    projections = np.array(projections, dtype=np.float32)
    
    proj_size = projections.shape[1]
    
    if initial_mrc_filename:
        with mrcfile.open(initial_mrc_filename) as mrcInitial:
            #vol_shape = list(map(int, mrcInitial.header.cella.tolist()[::-1]))
            vol_shape = [int(mrcInitial.header.nz), int(mrcInitial.header.ny), int(mrcInitial.header.nx)]
    else:
        vol_shape = [proj_size, proj_size, proj_size]

    print("Vol shape:", vol_shape)
        
    reconstruction = 0
    # Generate projs with ASTRA by batches 
    # Iter = int(len(projections_all)/batch_size) 
    # for i in range(Iter):
    #     projections = projections_all[i*batch_size : (i + 1)*batch_size, :, :]
    #     angles      = angles_all[i*batch_size : (i + 1)*batch_size, :]

    # Generate orientation vectors based on angles
    orientation_vectors   = RotationMatrix(angles)

    # Reshape projections correctly 
    projections1 = np.transpose(projections, (1, 0, 2))

    # Create projection 2D geometry in ASTRA
    proj_geom = astra.create_proj_geom('parallel3d_vec', proj_size, proj_size, orientation_vectors)
    projections_id = astra.data3d.create('-sino', proj_geom, projections1)

    # Create reconstruction.
    vol_geom = astra.creators.create_vol_geom(vol_shape[1], vol_shape[2], vol_shape[0])
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=reconstruction)

    alg_cfg = astra.astra_dict('CGLS3D_CUDA')  #  SIRT3D_CUDA
    alg_cfg['ProjectionDataId']     = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id, alg_iterations)
    reconstruction = astra.data3d.get(reconstruction_id)
    
    # Cleanup.
    astra.algorithm.delete(algorithm_id)
    astra.data3d.delete(reconstruction_id)
    astra.data3d.delete(projections_id)
        
    # Save reconstruction to mrc file for chimera
    if mrc_filename:
        Path(mrc_filename).parent.mkdir(parents=True, exist_ok=True)
        with mrcfile.new(mrc_filename, overwrite=overwrite) as mrc:
            if initial_mrc_filename:
                with mrcfile.open(initial_mrc_filename) as mrcInitial:
                    max_num = np.max(mrcInitial.data)
                    #print(max_num)

                    # Limit and scale reconstruction.
                    reconstruction[reconstruction < 0] = 0
                    reconstruction /= np.max(reconstruction)
                    #reconstruction = np.round(reconstruction * 255).astype(np.uint8)
                    reconstruction = reconstruction * max_num

                    #print(reconstruction.shape, mrcInitial.data.shape)
                    #print(np.array_equal(reconstruction, mrcInitial.data))
                    #print(np.max(reconstruction))

                    mrc.set_data(reconstruction)
                    mrc.header.cella = mrcInitial.header.cella
                    mrc.header.cellb = mrcInitial.header.cellb
                    mrc.header.origin= mrcInitial.header.origin
                    mrc.header.ispg  = mrcInitial.header.ispg
                    # mrc.header.nx = mrcInitial.header.nx
                    # mrc.header.ny = mrcInitial.header.ny
                    # mrc.header.nz = mrcInitial.header.nz
                    print(f"Reconstruction voxel size: {mrc.voxel_size}")
                    print(f"Initial voxel size: {mrcInitial.voxel_size}")
                    print("Header:")
                    print(f"\tmrc.header.cella: {mrc.header.cella}")
                    print(f"\tmrc.header.cellb: {mrc.header.cellb}")
                    print(f"\tmrc.header.origin: {mrc.header.origin}")
                    print(f"\tmrc.header.ispg: {mrc.header.ispg}")
            else:
                mrc.set_data(reconstruction)
    print(f"Reconstruction saved to: {mrc_filename}")        
    return reconstruction