
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
from cryoem.rotation_matrices import RotationMatrix
from tensorflow_graphics.geometry.transformation import quaternion
from cryoem.conversions import quaternion2euler

 
def generate_projections_ASTRA(Vol, Angles, ProjSize, BatchSizeAstra):
    """
    angles_gen_mode: str
        Takes values in [`uniform_angles`, `uniform_quaternions`]
    """
    # Create 3D geometry in ASTRA
    Vol_geom    = astra.create_vol_geom(Vol.shape[1], Vol.shape[2], Vol.shape[0])

    
    # Generate orientation vectors based on angles
    Orientation_Vectors   = RotationMatrix(Angles)

    # Create projection 2D geometry in ASTRA
    Proj_geom = astra.create_proj_geom('parallel3d_vec', ProjSize, ProjSize, Orientation_Vectors)

    # Generate projs 
    _, Proj_data = astra.create_sino3d_gpu(Vol, Proj_geom, Vol_geom)

    # Reshape projections correctly 
    Projections = np.transpose(Proj_data, (1, 0, 2))

    return Projections


def generate_2D_projections(input_file_path, ProjNber, AngCoverage, AngShift, Angles=None, angles_gen_mode=None, output_file_name=None, dtype=np.float32):
    """
    input_file_path: str
        Full path to the *.mrc file with 3D volume
    ProjNber: int
        Number of 2D projections 
    AngCoverage: list
        list of max values for each axis. E.g. `0.5,0.5,2.0` means it: x axis angle and y axis angle take values in range [0, 0.5*pi], z axis angles in range [0, 2.0*pi]
    AngShift: list
        Start of angular coverage
	angles_gen_mode: str
		2 options: (1) generate angles uniformly in angle space - 'uniform_angles', and (2) generate angles uniformly in quaternion space - 'uniform_S3'
    output_file_name: str
        Just the name of the output *.mat file. 
        If not specified, it will be generated automatically.
    """
    if AngCoverage is not None and AngShift is not None and Angles is not None:
        raise Exception("Please specify either AngCoverage and AngShift or just Angles")
    elif AngCoverage is not None and AngShift is not None and angles_gen_mode=='given':
        raise Exception("When AngCoverage and AngShift, the parameter angles_gen_mode cannot be `given`")
    elif Angles is None and angles_gen_mode=='given':
        raise Exception("When angles_gen_mode is `given` then Angles needs to be provided")

    # nber of projs created in a single ASTRA loop
    BatchSizeAstra = 50
    Vol = None 

    # filepaths 
    protein_name = input_file_path.split('/')[-1].split('.')[0]
    if angles_gen_mode != 'given':
        coverage_str = str(AngCoverage).replace(" ", "")[1:-1]
        shift_str    = str(AngShift).replace(" ", "")[1:-1]
        output_file_name = output_file_name or f'{protein_name}_ProjectionsAngles_ProjNber{ProjNber}_AngCoverage{coverage_str}_AngShift{shift_str}.h5'
    else:
        output_file_name = output_file_name or f'{protein_name}_Mode{angles_gen_mode}.h5'


    # get file extension
    extension = output_file_name.split('.')[-1]
    # storing output where the input mrc file is
    proj_ang_path = output_file_name  #os.path.join(os.path.dirname(input_file_path), output_file_name)
    get_directory = '/'.join(proj_ang_path.split("/")[:-1])
    pathlib.Path(get_directory).mkdir(parents=True, exist_ok=True)

    # loads data if data already exists 
    if os.path.exists(proj_ang_path):
        print('* Loading the dataset *')

        # read from the file
        if extension == "h5":
            with h5py.File(proj_ang_path, 'r') as data:
                Projections = dtype(data['Projections'])
                Angles      = dtype(data['Angles'])
        elif extension == "mat":
            with sio.loadmat(proj_ang_path) as data:
                Projections = dtype(data['Projections'])
                Angles      = dtype(data['Angles'])
        else: 
            raise NotImplementedError(f"Extension {extension} is not implemented")
    # generate data if data doesn't exist  
    else:
        print('* Generating the dataset *')

        # Load 3D volume
        # Value error fix explained here: https://mrcfile.readthedocs.io/en/latest/usage_guide.html 
        try:
            with mrcfile.open(input_file_path) as mrcVol:
                Vol      = np.array(mrcVol.data) 
                ProjSize = int(np.sqrt(np.sum(np.square(Vol.shape))))
        except ValueError:
            with mrcfile.open(input_file_path, mode='r+', permissive=True) as mrcVol:
                mrcVol.header.map = mrcfile.constants.MAP_ID
                Vol      = np.array(mrcVol.data) 
                ProjSize = int(np.sqrt(np.sum(np.square(Vol.shape))))

        # Initialisations 
        Projections = np.zeros((ProjNber, ProjSize, ProjSize), dtype=dtype)

        if Angles is None:
            Angles = np.zeros((ProjNber, 3), dtype=dtype)

            # Generate random angles
            if angles_gen_mode == "uniform_angles":
                Z1 =  AngShift[0]*np.pi + AngCoverage[0]*np.pi*np.random.random(size=(ProjNber, 1))
                Y2 =  AngShift[1]*np.pi + AngCoverage[1]*np.pi*np.random.random(size=(ProjNber, 1))
                Z3 =  AngShift[2]*np.pi + AngCoverage[2]*np.pi*np.random.random(size=(ProjNber, 1))
                Angles = np.concatenate((Z1, Y2, Z3), axis=1)
            elif angles_gen_mode == "uniform_S2":
                compensation =  10 * 2/(AngCoverage[1]*AngCoverage[2]) 
                #print("compensation", compensation)
                quaternions = quaternion.normalized_random_uniform(quaternion_shape=(int(compensation*ProjNber),))
                Angles = quaternion2euler(quaternions).numpy()
                
                for i, a in enumerate(Angles):
                    Angles[i] = [a[0]%(2*np.pi), a[1]%(np.pi), a[2]%(2*np.pi)]
                
                indices = np.where((AngShift[0]*np.pi<=Angles[:,0]) & (Angles[:,0]<=AngShift[0]*np.pi+AngCoverage[0]*np.pi) & \
                                (AngShift[1]*np.pi<=Angles[:,1]) & (Angles[:,1]<=AngShift[1]*np.pi+AngCoverage[1]*np.pi) & \
                                (AngShift[2]*np.pi<=Angles[:,2]) & (Angles[:,2]<=AngShift[2]*np.pi+AngCoverage[2]*np.pi))[0]
                
                indices = indices[:ProjNber]
                Angles = np.take(Angles, indices, axis=0)
            else:
                raise NotImplemented(f"This coverage is not implemented yet - {angles_gen_mode}!")
        print(Angles.shape)

        # Explicitly putting the first set of angles to 0,0,0 for future debugging if needed
        Angles[0] = [0, 0, 0]

        # Generate projs with ASTRA by batches 
        Iter = int(ProjNber/BatchSizeAstra) 
        for i in range(Iter):

            # Generate projections 
            projections = generate_projections_ASTRA(Vol, Angles[i*BatchSizeAstra : (i + 1)*BatchSizeAstra], ProjSize, BatchSizeAstra)

            # Concatenate generated projections 
            Projections[i*BatchSizeAstra : (i + 1)*BatchSizeAstra, :, :] = projections

        Projections = np.array(Projections, dtype=dtype)
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

    print(f"Protein:         {protein_name}")
    print(f"Input filename:  {input_file_path}")
    print(f"Output filename: {output_file_name}")
    if Vol is not None: print(f"Volume:          {Vol.shape}")
    print(f'Projections (#): {Projections.shape}')
    print(f'Angles (#):      {Angles.shape}\n')
    print('*'*10)