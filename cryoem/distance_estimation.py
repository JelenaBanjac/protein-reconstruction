
import os
import h5py
from time import time, strftime
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pathlib
from cryoem.rotation_matrices import RotationMatrix
from cryoem.conversions import euler2quaternion, d_q
from cryoem.knn import get_knn_projections

import random
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AvgPool2D, Lambda
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

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

def sample_pairs(projections, num_pairs, style="random", k=None):
    if not k and style != "random":
        raise ValueError("Please specify k for kNN for sample_pairs method")
    
    if style=="random":
        idx1 = list(np.random.choice(projections, size=num_pairs))
        idx2 = list(np.random.choice(projections, size=num_pairs))
    
    elif style=="knn":
        idx1 = list(np.random.choice(projections, size=num_pairs))
        indices_p, distances_p, A_p = get_knn_projections(k=k)
        idx2 = [indices_p[i][np.random.randint(1, k)] for i in idx1]
 
    elif style=="knn_and_random":
        # select random sample for the first element of pair
        idx1 = list(np.random.choice(projections, size=num_pairs))
        
        # half from kNN
        indices_p, distances_p, A_p = get_knn_projections(k=k)
        idx2_knn = [indices_p[i][np.random.randint(1, k)] for i in idx1[:num_pairs//2]]
        idx2_random = list(np.random.randint(0, num_projections, num_pairs//2))
        # half random
        idx2 = idx2_knn + idx2_random
        
    return idx1, idx2


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def create_pairs(x, y, indices, num_pairs):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    
    # Sample some pairs.
    idx1, idx2 = sample_pairs(projections=indices, num_pairs=num_pairs, style="random")
    
    for z1, z2 in zip(idx1, idx2):
        pairs += [[x[z1], x[z2]]]
        labels += [d_q(euler2quaternion(y[z1]), euler2quaternion(y[z2]))]

    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input_x = Input(shape=input_shape)
    #print(input_shape)

    # add Convolution, MaxPool, Conv2D, remove Dropout and Dense
    x = Conv2D(filters=16, kernel_size=[9, 9], activation='relu', padding='same', kernel_initializer='glorot_uniform')(input_x)
    x = MaxPooling2D([2, 2], padding='same')(x)

    x = Conv2D(filters=32, kernel_size=[7, 7], activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
    x = MaxPooling2D([2, 2], padding='same')(x)

    x = Conv2D(64, [5, 5], activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
    x = MaxPooling2D([2, 2], padding='same')(x)

    x = Conv2D(128, [3, 3], activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
    x = MaxPooling2D([2, 2], padding='same')(x)

    x = Conv2D(256, [1, 1], activation='relu', padding='same', kernel_initializer='glorot_uniform')(x)
    x = MaxPooling2D([2, 2], padding='same')(x)

    if input_shape[0] == 116: size = [8, 8]
    elif input_shape[0] == 275: size = [24,24]
    else: print("Put pool size")
    x = AvgPool2D(pool_size=size, padding='same')(x)

    x = tf.squeeze(x, axis=[1,2])
    
    return Model(input_x, x)


def train_siamese(training_pairs, training_y, validation_pairs, validation_y, epochs, batch_size, learning_rate, plot=True):
    input_shape = training_pairs[:, 0].shape[1:]
    print(f"Input images shape {input_shape}")

    # network definition
    base_network = create_base_network(input_shape)


    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)


    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)



    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)


    # train
    #optimizer = RMSprop()
    optimizer = Adam(learning_rate=learning_rate)

    #model.compile(loss=mse, optimizer=optimizer, metrics=['mae'])
    model.compile(loss=mae, optimizer=optimizer, metrics=['mse'])


    model.summary()

    plot_model(model, to_file="figures/model_plot.png", show_shapes=True, show_layer_names=True)

    # Create a callback that saves the model's weights
    CHECKPOINT_PATH = f"training/{strftime('%Y%m%d_%H%M%S')}"
    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    backup_callback = ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                      save_weights_only=True,
                                      verbose=1)
    # Create a callback that will show tensorboard data
    LOGS_PATH = f"logs/{strftime('%Y%m%d_%H%M%S')}"
    pathlib.Path(LOGS_PATH).mkdir(parents=True, exist_ok=True)
    logs_callback = TensorBoard(LOGS_PATH, histogram_freq=1)

    history = model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=([validation_pairs[:, 0], validation_pairs[:, 1]], validation_y),
                    callbacks=[backup_callback, logs_callback])

    model_filename = f"training/{strftime('%Y%m%d_%H%M%S')}.h5"
    model.save(model_filename) 
    print(f"Model saved to: {model_filename}")

    if plot:
        # Get training and test loss histories
        training_loss = history.history['loss']
        val_loss = history.history['val_loss']
        mses = history.history['mse']
        val_mses = history.history['val_mse']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))

        # Create count of the number of epochs
        epoch_count = range(1, len(training_loss) + 1)

        # Visualize loss history
        ax1.plot(epoch_count, training_loss, 'r--', label='MAE Training Loss')
        ax1.plot(epoch_count, val_loss, 'b-', label='MAE Validation Loss')
        ax1.legend()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(epoch_count, mses, 'r-', label='MSE Training')
        ax2.plot(epoch_count, val_mses, 'b-', label='MSE Validation')
        ax2.legend()
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        plt.show();

    return model, history

def plot_results(projections, y_pred, y, strtype):
    if projections.shape[-1] == 1:
        projections = projections.reshape(list(projections.shape[:-2]) +[-1])

    def _inner(i):
        
        plt.imfig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(projections[i, 0])
        ax2.imshow(projections[i, 1])

        print(f"--- {strtype} Set ---")
        print(f"predicted: {y_pred[i][0]}")
        print(f"true:      {y[i].numpy()}")
        print(f"mse:       {mse(y_pred[i], y[i].numpy())}")
        print(f"mae:       {mae(y_pred[i], y[i].numpy())}")
        
    return _inner
