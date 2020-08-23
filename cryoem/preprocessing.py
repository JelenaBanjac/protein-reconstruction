import numpy as np
from sklearn.model_selection import train_test_split
import os
import math

def global_standardization(X):
    """Does not have all the positive piels
    Ref: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/""" 
    print(f'Image shape: {X[0].shape}')
    print(f'Data Type: {X[0].dtype}')
    X = X.astype('float32')

    print("***")
    ## GLOBAL STANDARDIZATION
    # calculate global mean and standard deviation
    mean, std = X.mean(), X.std()
    print(f'Mean: {mean:.3f} | Std: {std:.3f}')
    print(f'Min:  {X.min():.3f} | Max: {X.max():.3f}')
    # global standardization of pixels
    X = (X - mean) / std
    # confirm it had the desired effect
    mean, std = X.mean(), X.std()
    print(f'Mean: {mean:.3f} | Std: {std:.3f}')
    print(f'Min:  {X.min():.3f} | Max: {X.max():.3f}')
    
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

def rescale_images(original_images):
    mobile_net_possible_dims = [128, 160, 192, 224]
    dim_goal = 128
    
    for dim in mobile_net_possible_dims:
        if original_images.shape[1] <= dim:
            dim_goal = dim
            break;
    print(f"Image rescaled from dimension {original_images.shape[1]} to {dim_goal} for MobileNet")
    scale = dim_goal/original_images.shape[1]
    images = np.empty((original_images.shape[0], dim_goal, dim_goal))
    for i, original_image in enumerate(original_images):
        images[i] = rescale(original_image, (scale, scale), multichannel=False)
    return images

def gaussian_noise(shape, mean=0, var=0.1): 
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (shape[0], shape[1]))
    gauss = gauss.reshape(shape[0], shape[1])
    return gauss

def add_gaussian_noise(projections, noise_var):
    """
    projections = add_gaussian_noise(projections, NOISY_VAR)
    """
    noise_sigma   = noise_var**0.5
    nproj,row,col = projections.shape
    gauss_noise   = np.random.normal(0,noise_sigma,(nproj,row,col))
    gauss_noise   = gauss_noise.reshape(nproj,row,col) 
    projections   = projections + gauss_noise
    return projections

def add_triangle_translation(projections, left_limit, peak_limit, right_limit):
    """
    projections = add_triangle_translation(projections, left_limit=-TRANSLATION, peak_limit=0, right_limit=TRANSLATION)
    """
    horizontal_shift = np.random.triangular(left_limit, peak_limit, right_limit, len(projections))
    vertical_shift   = np.random.triangular(left_limit, peak_limit, right_limit, len(projections))
    for i, (hs, vs) in enumerate(zip(horizontal_shift, vertical_shift)):
        # shift 1 place in horizontal axis
        projections[i] = np.roll(projections[i], int(hs), axis=0)
        # shift 1 place in vertical axis
        projections[i] = np.roll(projections[i], int(vs), axis=1) 
    return projections

def channels_setup(X, channels="gray"):
    if channels == "rgb":
        X = np.stack((X,)*3, axis=-1)
    elif channels == "gray":
        X = X[:,:,:,np.newaxis]

    return X

def preprocessing(projections, noise_var, left_limit, peak_limit, right_limit, channels):
    # add gaussian noise
    projections = add_gaussian_noise(projections, noise_var)

    # add translation
    projections = add_triangle_translation(projections, left_limit=left_limit, peak_limit=peak_limit, right_limit=right_limit)

    # normalize pixel values
    projections = global_standardization(projections)

    # rgb or gray scale images
    projections = channels_setup(projections, channels)

    return projections

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
