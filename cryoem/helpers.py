
import numpy as np
import os
from sklearn.model_selection import train_test_split

def train_val_test_split(indices, file_name):
    """Train-validation-test split of indices"""
    if not os.path.exists(file_name):
        # the data, split between train and test sets
        train_idx, test_idx = train_test_split(indices, 
                                               test_size=0.33, 
                                               random_state=42)
        train_idx, val_idx= train_test_split(train_idx, 
                                             test_size=0.25, 
                                             random_state=1)

        train_idx = sorted(train_idx)
        val_idx = sorted(val_idx)
        test_idx = sorted(test_idx)

        np.savez(file_name, train_idx, val_idx, test_idx)
    else:
        data = np.load(file_name)
        train_idx, val_idx, test_idx = data["arr_0"], data["arr_1"], data["arr_2"]
        
    return train_idx, val_idx, test_idx

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
    """Rescale the protein images"""
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


def add_gaussian_noise(projections, noise_var):
    """Add Gaussian noise to the protein projection image"""
    noise_sigma   = noise_var**0.5
    nproj,row,col = projections.shape
    gauss_noise   = np.random.normal(0, noise_sigma, (nproj, row, col))
    gauss_noise   = gauss_noise.reshape(nproj, row, col) 
    projections   = projections + gauss_noise
    return projections

def add_triangle_translation(projections, left_limit, peak_limit, right_limit):
    """Add triangular distribution shift to protein center"""
    horizontal_shift = np.random.triangular(left_limit, peak_limit, right_limit, len(projections))
    vertical_shift   = np.random.triangular(left_limit, peak_limit, right_limit, len(projections))
    for i, (hs, vs) in enumerate(zip(horizontal_shift, vertical_shift)):
        projections[i] = np.roll(projections[i], int(hs), axis=0) # shift 1 place in horizontal axis
        projections[i] = np.roll(projections[i], int(vs), axis=1) # shift 1 place in vertical axis
    return projections

def projections_preprocessing(projections, angles_true, settings=None):
    """Collection of projection's preprocessing"""
    
    settings_default = dict(
        noise={"variance":0.0},
        shift={"left_limit":-0.01,
               "peak_limit":0,
               "right_limit":0.01},
        channels="gray")
    if settings is None:
        settings = {}
    settings_final = {**settings_default, **settings}
    
    projections = add_gaussian_noise(projections, settings_final["noise"]["variance"])
    projections = add_triangle_translation(projections, left_limit=settings_final["shift"]["left_limit"], peak_limit=settings_final["shift"]["peak_limit"], right_limit=settings_final["shift"]["right_limit"])
    
    X, y = np.array(projections, dtype=np.float32), np.array(angles_true, dtype=np.float32)
    X = global_standardization(X)
    
    if settings_final["channels"] == "rgb":
        X = np.stack((X,)*3, axis=-1)
    elif settings_final["channels"] == "gray":
        X = X[:,:,:,np.newaxis]
        
    return X, y