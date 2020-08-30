import numpy as np
from sklearn.model_selection import train_test_split
import os
import math
from skimage.transform import rescale

def global_standardization(X):
    """Does not have all the positive piels
    Ref: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/""" 
    print("Global standardization")
    print(f'\tImage shape: {X[0].shape}')
    print(f'\tData Type: {X[0].dtype}')
    X = X.astype('float32')

    # calculate global mean and standard deviation
    mean, std = X.mean(), X.std()
    print(f'\tMean: {mean:.3f} | Std: {std:.3f}')
    print(f'\tMin:  {X.min():.3f} | Max: {X.max():.3f}')
    # global standardization of pixels
    X = (X - mean) / std
    # confirm it had the desired effect
    mean, std = X.mean(), X.std()
    print(f'\tMean: {mean:.3f} | Std: {std:.3f}')
    print(f'\tMin:  {X.min():.3f} | Max: {X.max():.3f}')
    
    return X

# TODO: check if needed
def positive_global_standardization(X):
    """Has all positive pixels
    Ref: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/"""
    mean, std = X.mean(), X.std()
    print(f"Mean: {mean:.3f} | Std: {std:.3f}")

    # global standardization of pixels
    X = (X - mean) / std

    # clip pixel values to [-1,1]
    X = np.clip(X, -1.0, 1.0)

    # shift from [-1,1] to [0,1] with 0.5 mean
    X = (X + 1.0) / 2.0

    # confirm it had the desired effect
    mean, std = X.mean(), X.std()
    print(f'Mean: {mean:.3f} | Std: {std:.3f}')
    print(f'Min:  {X.min():.3f} | Max: {X.max():.3f}')
    
    return X

def rescale_images(original_images, rescale_dim=128):
    mobile_net_possible_dims = [128, 160, 192, 224]
    #rescale_dim = 128

    original_images = np.array(original_images)
    #print(original_images.shape)
    #print(type(original_images))
    
#     for dim in mobile_net_possible_dims:
#         if original_images.shape[1] <= dim:
#             rescale_dim = dim
#             break;
    
    scale = rescale_dim/original_images.shape[1]
    images = np.empty((original_images.shape[0], rescale_dim, rescale_dim))
    for i, original_image in enumerate(original_images):
        images[i] = rescale(original_image, (scale, scale), multichannel=False)
    print(f"Image rescaled: from dimension {original_images.shape[1]} to {rescale_dim}")
    return images

# def gaussian_noise(shape, mean=0, var=0.1): 
#     sigma = var**0.5
#     gauss = np.random.normal(mean, sigma, (shape[0], shape[1]))
#     gauss = gauss.reshape(shape[0], shape[1])
#     return gauss

def add_gaussian_noise(projections, noise_var):
    """
    projections = add_gaussian_noise(projections, NOISY_VAR)
    """
    print("Noise:", sep=" ")
    print("Variance=", noise_var)
    if noise_var==0:
        print("No noise")
        return projections
    noise_sigma   = noise_var**0.5
    gauss_noise   = np.random.normal(0, noise_sigma, projections.shape)
    gauss_noise   = gauss_noise.reshape(*projections.shape) 
    projections   = projections + gauss_noise

    return projections

def add_triangle_translation(projections, left_limit, peak_limit, right_limit):
    """
    projections = add_triangle_translation(projections, left_limit=-TRANSLATION, peak_limit=0, right_limit=TRANSLATION)
    """
    print("Translation:", sep=" ")
    if left_limit==0 and right_limit==0:
        print("No translation")
        return projections
    horizontal_shift = np.random.triangular(left_limit, peak_limit, right_limit, len(projections))
    vertical_shift   = np.random.triangular(left_limit, peak_limit, right_limit, len(projections))
    for i, (hs, vs) in enumerate(zip(horizontal_shift, vertical_shift)):
        # shift 1 place in horizontal axis
        projections[i] = np.roll(projections[i], int(hs), axis=0)
        # shift 1 place in vertical axis
        projections[i] = np.roll(projections[i], int(vs), axis=1) 
    print(f"left_limit={left_limit}, peak_limit={peak_limit}, right_limit={right_limit}")
    return projections

def channels_setup(X, channels=1):
    if channels == 3: # rgb images
        X = np.stack((X,)*3, axis=-1)
    elif channels == 1: # gray-scale 
        X = X[:,:,:,np.newaxis]

    return X

def preprocessing(projections, PROJECTIONS_NUM_SINGLE, rescale_dim, noise_var_scale, left_limit, peak_limit, right_limit, channels):
    print("--- Preprocessing projections ---")
    projections_new = np.empty((len(projections), rescale_dim, rescale_dim, channels))
    
    for i in range(0, len(projections), PROJECTIONS_NUM_SINGLE):
        print("Protein #", i//PROJECTIONS_NUM_SINGLE+1)
        protein_projections = np.empty((PROJECTIONS_NUM_SINGLE, *projections[i].shape))
        for j in range(PROJECTIONS_NUM_SINGLE):
            protein_projections[j] = projections[i+j]
            
        protein_projections = rescale_images(protein_projections, rescale_dim)

        
        # normalize pixel values
        protein_projections = global_standardization(protein_projections)
        
        # add gaussian noise
        protein_projections = add_gaussian_noise(protein_projections, noise_var=noise_var_scale*np.max(protein_projections))

        # add translation
        protein_projections = add_triangle_translation(protein_projections, left_limit=left_limit, peak_limit=peak_limit, right_limit=right_limit)
        
        # rgb or gray scale images
        protein_projections = channels_setup(protein_projections, channels)

        projections_new[i:i+PROJECTIONS_NUM_SINGLE] = protein_projections
        
    return projections_new

def train_val_test_split(projections_num, test_size=0.33, val_size=0.25, train_percent=0.01, val_percent=0.01, indices_file="../data/train_val_test_indices.npz"):
    if not os.path.exists(indices_file):
        train_idx, test_idx = train_test_split(range(projections_num), test_size=test_size)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size)
        np.savez(indices_file, train_idx, val_idx, test_idx)
    else:
        data = np.load(indices_file)
        train_idx, val_idx, test_idx = data["arr_0"], data["arr_1"], data["arr_2"]
    
    print(f"TRAIN: {1-test_size:.2f} x {1-val_size:.2f} = {(1-test_size)*(1-val_size):.2f} => {str(len(train_idx)).rjust(5)} imgs => max pairs: {str(len(train_idx)**2).rjust(10)}   |   {int(train_percent*np.power(len(train_idx), 2))}")
    print(f"TEST : {str(test_size).rjust(18)} => {str(len(test_idx)).rjust(5)} imgs => max pairs: {str(len(test_idx)**2).rjust(10)}   |   all")
    print(f"VAL  : {1-test_size:.2f} x {val_size:.2f} = {(1-test_size)*val_size:.2f} => {str(len(val_idx)).rjust(5)} imgs => max pairs: {str(len(val_idx)**2).rjust(10)}   |   {int(val_percent*np.power(len(val_idx), 2))}")
    print(f"Indices stored in {indices_file}")

    train_pairs_num = int(train_percent*np.power(len(train_idx), 2))
    val_pairs_num = int(val_percent*np.power(len(val_idx), 2))

    return train_idx, val_idx, test_idx, train_pairs_num, val_pairs_num
