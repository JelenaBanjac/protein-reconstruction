
"""""
@author: fangshu.yang@epfl.ch, laurene.donati@epfl.ch 
"""

import numpy as np
import random
import os, sys
import scipy.io as sio
sys.path.append(os.getcwd())
import time
import mrcfile
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import astra
import pathlib
import h5py


def RotationMatrix(angles):
	"""
	Rotation matrix from: https://www.geometrictools.com/Documentation/EulerAngles.pdf
	Chapter 2.8. Factor as Rx0 Rz Rx1
	Also, playing with: https://eater.net/quaternions/video/doublecover
	"""
	#print(angles.shape)

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
	 
	R = np.concatenate([np.concatenate([c2,    -c3*s2,           s2*s3            ], axis=2), 
						np.concatenate([c1*s2, c1*c2*c3 - s1*s3, -c3*s1 - c1*c2*s3], axis=2),
						np.concatenate([s1*s2, c1*s3 + c2*c3*s1, c1*c3 - c2*s1*s3 ], axis=2)],
						axis=1)

	# rotate previous values
	vectors[:,0:3] = np.matmul(R,vector[0:3])
	vectors[:,6:9] = np.matmul(R,vector[6:9])
	vectors[:,9:12] = np.matmul(R,vector[9:12])
	
	return vectors


def project_volume(Vol, Angles, Vol_geom, ProjSize):
		 
	# Generate orientation vectors based on angles
	Orientation_Vectors   = RotationMatrix(Angles)
	
	# Create projection 2D geometry in ASTRA
	Proj_geom = astra.create_proj_geom('parallel3d_vec', ProjSize, ProjSize, Orientation_Vectors)
	
	# Generate projs 
	_, Proj_data = astra.create_sino3d_gpu(Vol, Proj_geom, Vol_geom)
	
	# Reshape projections correctly 
	Projections = np.transpose(Proj_data, (1, 0, 2))
	
	return  Projections

 
def gen_projs_ASTRA(Vol, AngCoverage, AngShift, ProjSize, BatchSizeAstra): 
		
	# Create 3D geometry in ASTRA
	Vol_geom    = astra.create_vol_geom(Vol.shape[1], Vol.shape[2], Vol.shape[0])
	
	# Generate random angles
	Angles      = AngShift + AngCoverage*2*np.pi*np.random.random(size=(BatchSizeAstra, 3))
	
	# Generate projections
	Projections = project_volume(Vol, Angles, Vol_geom, ProjSize)
	
	return Projections, Angles


def generate_2D_projections(input_file_path, ProjNber, AngCoverage, AngShift, output_file_name=None):
	"""
	input_file_path: str
		Full path to the *.mrc file with 3D volume
	ProjNber: int
		Number of 2D projections 
	AngCoverage: float
		Angular coverage (0.5: half-sphere, 1: complete sphere) 
	AngShift: np.float 
		Start of angular coverage
	output_file_name: str
		Just the name of the output *.mat file. 
		If not specified, it will be generated automatically.
	"""
	# nber of projs created in a single ASTRA loop
	BatchSizeAstra = 50 
	
	# filepaths 
	output_file_name = output_file_name or f'ProjectionsAngles_ProjNber{ProjNber}_AngCoverage{AngCoverage}_AngShift{AngShift:.2f}.h5'
	# get file extension
	extension = output_file_name.split('.')[-1]
	# storing output where the input mrc file is
	proj_ang_path = os.path.join(os.path.dirname(input_file_path), output_file_name)
	
	# loads data if data already exists 
	if os.path.exists(proj_ang_path):
		print('* Loading the dataset *\n')
	
		# read from the file
		if extension == "h5":
			with h5py.File(proj_ang_path, 'r') as data:
				Projections = np.float32(data['Projections'])
				Angles      = np.float32(data['Angles'])
		elif extension == "mat":
			with sio.loadmat(proj_ang_path) as data:
				Projections = np.float32(data['Projections'])
				Angles      = np.float32(data['Angles'])
		else: 
			raise NotImplementedError(f"Extension {extension} is not implemented")

	# generate data if data doesn't exist  
	else:
		print('* Generating the dataset *\n')
	
		# Load 3D volume 
		with mrcfile.open(input_file_path) as mrcVol:
			Vol      = np.array(mrcVol.data) 
			ProjSize = int(np.sqrt(np.sum(np.square(Vol.shape))))
		
		# Initialisations 
		Projections = np.zeros((ProjNber, ProjSize, ProjSize), dtype=float)
		Angles      = np.zeros((ProjNber, 3), dtype=float)
		
		# Generate projs with ASTRA by batches 
		Iter = int(ProjNber/BatchSizeAstra) 
		for i in range(Iter):
			
			# Generate projections 
			projections, angles = gen_projs_ASTRA(Vol, AngCoverage, AngShift, ProjSize, BatchSizeAstra)
			
			# Concatenate generated projections 
			Projections[i*BatchSizeAstra : (i + 1)*BatchSizeAstra, :, :] = projections
			Angles[i*BatchSizeAstra : (i + 1)*BatchSizeAstra, :] = angles  
			
		# Save data 
		if extension == "h5":
			with h5py.File(proj_ang_path, 'w') as hf:
				hf.create_dataset('Projections', data=Projections)
				hf.create_dataset('Angles', data=Angles)
		elif extension == "mat":
			sio.savemat(proj_ang_path, {'Projections': Projections, 
										'Angles': Angles}) 
		else: 
			raise NotImplementedError(f"Extension {extension} is not implemented")
		
	print(f'Projections: {Projections.shape}')
	print(f'Angles: {Angles.shape}\n')
