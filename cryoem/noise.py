import numpy as np

def gaussian_noise(shape, mean=0, var=0.1): 
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (shape[0], shape[1]))
    gauss = gauss.reshape(shape[0], shape[1])
    return gauss