import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import imread, imwrite
 
import astra
 

data = np.load('../data/5a1a_projections_and_angles.npz')
projections, angles = data["arr_0"], data["arr_1"]



# Configuration.
distance_source_origin = 300  # [mm]
distance_origin_detector = 100  # [mm]
detector_pixel_size = 1.05  # [mm]
detector_rows = 200  # Vertical size of detector [pixels].
detector_cols = 200  # Horizontal size of detector [pixels].
num_of_projections = 180
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
input_dir = 'dataset'
output_dir = 'reconstruction'
 
# Load projections.
projections = np.zeros((detector_rows, num_of_projections, detector_cols))
for i in range(num_of_projections):
    im = imread(join(input_dir, 'proj%04d.tif' % i)).astype(float)
    im /= 65535
    projections[:, i, :] = im
 
# Copy projection images into ASTRA Toolbox.
proj_geom = \
  astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                         (distance_source_origin + distance_origin_detector) /
                         detector_pixel_size, 0)
projections_id = astra.data3d.create('-sino', proj_geom, projections)
 
# Create reconstruction.
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                          detector_rows)
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
alg_cfg = astra.astra_dict('FDK_CUDA')
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
