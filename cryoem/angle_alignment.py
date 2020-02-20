from cryoem.quaternions import euler2quaternion, d_q, quaternion2euler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_graphics.geometry.transformation import quaternion
from time import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)
import random
from tensorflow_graphics.math import vector
from cryoem.angle_recovery import geodesic_distance
from itertools import product



def lossR(a_R, a_predicted, a_true):
    q_predicted = euler2quaternion(a_predicted)
    q_true = euler2quaternion(a_true)
    q_R = euler2quaternion(a_R)

    rotated_quaternion = quaternion.multiply(q_predicted, q_R)
    distance = d_q(q_true, rotated_quaternion)
    
    return tf.reduce_mean(distance)

def gradientR(a_R, a_predicted, a_true):
    with tf.GradientTape() as tape:
        loss_value = lossR(a_R, a_predicted, a_true)
        gradient = tape.gradient(loss_value, a_R)
        
    return loss_value, gradient

def training_angle_alignment_R_flips(steps, batch_size, projection_idx, learning_rate, angles_true, ap, optimization=True):
    flips = {}

    # (1, -1, 1) -> loss
    for z0_flip, y1_flip, z2_flip in list(product([1, -1], repeat=3)):
    
        print(f"FLIPPING: {z0_flip, y1_flip, z2_flip}")

        ap_new = np.zeros(ap.shape)
        for i, a in enumerate(ap):
            ap_new[i] = [z0_flip*a[0], y1_flip*a[1], z2_flip*a[2]]
            
        angles_predicted = tf.convert_to_tensor(ap_new)

        a_R, loss = training_angle_alignment_R(steps=steps, 
                                    batch_size=batch_size,
                                    projection_idx=projection_idx,
                                    learning_rate=learning_rate,
                                    angles_true=angles_true,
                                    angles_predicted=angles_predicted,
                                    optimization=True)
        
        print(f"Rotation: {a_R[0].numpy()};  Loss: {loss}")
        flips[(z0_flip, y1_flip, z2_flip)] = (a_R, loss)
    
    best_flips = min(flips, key=lambda x: flips.get(x)[1])
    return best_flips, flips[best_flips]

def updateR_alignment(flips, ap, a_R):
    ap_new = np.zeros(ap.shape)
    for i, a in enumerate(ap):
        ap_new[i] = [flips[0]*a[0], flips[1]*a[1], flips[2]*a[2]]

    angles_predicted = tf.convert_to_tensor(ap_new)

    q_predicted = euler2quaternion(angles_predicted)
    q_R = euler2quaternion(a_R)

    angles_predicted_new = quaternion2euler(quaternion.multiply(q_predicted, q_R))
    return angles_predicted_new


def training_angle_alignment_R(steps, batch_size, projection_idx, learning_rate, angles_true, angles_predicted, optimization=True):
    optimizer = Adam(learning_rate=learning_rate)

    losses = np.empty(steps)
    time_start = time()

    euler = np.zeros(3, dtype=np.float64)
    a_R = [tf.Variable(euler)]

    for step in range(1, steps+1):

        # Sample some pairs.
        idx = list(np.random.choice(projection_idx, size=batch_size))
        
        # Compute distances between projections
        a_true = [angles_true[i] for i in idx]
        a_predicted = [angles_predicted[i] for i in idx]
        
        # Optimize by gradient descent.
        if optimization:
            losses[step-1], gradients = gradientR(a_R, a_predicted, a_true)
            optimizer.apply_gradients(zip(gradients, a_R))
        else:
            losses[step-1] = lossR(a_R, a_predicted, a_true)

        # Periodically report progress.
        if ((step % (steps//10)) == 0) or (step == steps):
            time_elapsed = time() - time_start
            print(f'step {step}/{steps} ({time_elapsed:.0f}s): loss = {losses[step-1]:.2e}')

    if optimization:
        # Plot convergence.
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, time()-time_start, steps), losses)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('loss');
    else:
        print(f"Mean distance difference: {np.mean(losses)}")

    return a_R, losses[-1]

def RotationMatrix(angles):
    # create rotation matrix
    c1 = tf.reshape(tf.cos(angles[:,0]), (-1,1,1))
    c2 = tf.reshape(tf.cos(angles[:,1]), (-1,1,1))
    c3 = tf.reshape(tf.cos(angles[:,2]), (-1,1,1))

    s1 = tf.reshape(tf.sin(angles[:,0]), (-1,1,1))
    s2 = tf.reshape(tf.sin(angles[:,1]), (-1,1,1))
    s3 = tf.reshape(tf.sin(angles[:,2]), (-1,1,1))
    
    # Euler angles
    # R = np.concatenate([np.concatenate([c3*c2*c1-s3*s1, c3*c2*s1 + s3*c1, -c3*s2],axis=2),\
    # 				np.concatenate([-s3*c2*c1-c3*s1,-s3*c2*s1+c3*c1 , s3*s2],axis=2),\
    # 				np.concatenate( [s2*c1,          s2*s1          , c2],axis=2)],axis=1)
    R = tf.concat([tf.concat([c1*c2*c3-s1*s3, c1*s3+c2*c3*s1 , -c3*s2],axis=2),\
                    tf.concat([-c3*s1-c1*c2*s3,    c1*c3-c2*s1*s3 ,   s2*s3],axis=2),\
                    tf.concat( [c1*s2,             s1*s2          ,   c2],axis=2)],axis=1)
    # BT angles
    # R = np.concatenate([np.concatenate([c1*c2, c2*s1, -s2],axis=2),\
    # 				np.concatenate([c1*s2*s3-c3*s1, c1*c3+s1*s2*s3, c2*s3],axis=2),\
    # 				np.concatenate( [s1*s3+c1*c3*s2, c3*s1*s2-c1*s3, c2*c3],axis=2)],axis=1)

    vec03 = tf.expand_dims(tf.transpose(tf.convert_to_tensor([0,0,1], dtype=tf.float64)), 1)
    vec36 = tf.expand_dims(tf.transpose(tf.convert_to_tensor([0,0,0], dtype=tf.float64)), 1)
    vec69 = tf.expand_dims(tf.transpose(tf.convert_to_tensor([1,0,0], dtype=tf.float64)), 1)
    vec912 = tf.expand_dims(tf.transpose(tf.convert_to_tensor([0,1,0], dtype=tf.float64)), 1)
    
    vec_all03 = tf.squeeze(tf.matmul(R, vec03))
    vec_all36 = tf.squeeze(tf.matmul(R, vec36))
    vec_all69 = tf.squeeze(tf.matmul(R, vec69))
    vec_all912 = tf.squeeze(tf.matmul(R, vec912))

    vectors = tf.concat([vec_all03, vec_all36, vec_all69, vec_all912], axis=1)

    return vectors

def lossGD(a_R, a_predicted, a_true):
    a_predicted = tf.math.add(a_predicted, a_R)

    a_true = tf.convert_to_tensor(a_true)
    aps = RotationMatrix(a_predicted)[:,:3]
    ats = RotationMatrix(a_true)[:,:3]
    
    gd = geodesic_distance(ats, aps)

    # The mean doesn't depend on the batch size.
    return gd

def gradientGD(a_R, a_predicted, a_true):
    with tf.GradientTape() as tape:
        loss_value = lossGD(a_R, a_predicted, a_true)
        gradient = tape.gradient(loss_value, a_R)
        
    return loss_value, gradient

def training_angle_alignment_GD_flips(steps, batch_size, projection_idx, learning_rate, angles_true, ap, optimization=True):
    flips = {}

    # (1, -1, 1) -> loss
    for z0_flip, y1_flip, z2_flip in list(product([1, -1], repeat=3)):
    
        print(f"FLIPPING: {z0_flip, y1_flip, z2_flip}")

        ap_new = np.zeros(ap.shape)
        for i, a in enumerate(ap):
            ap_new[i] = [z0_flip*a[0], y1_flip*a[1], z2_flip*a[2]]
            
        angles_predicted = tf.convert_to_tensor(ap_new)

        a_R, loss = training_angle_alignment_GD(steps=steps, 
                                    batch_size=batch_size,
                                    projection_idx=projection_idx,
                                    learning_rate=learning_rate,
                                    angles_true=angles_true,
                                    angles_predicted=angles_predicted,
                                    optimization=True)
        
        print(f"Rotation: {a_R[0].numpy()};  Loss: {loss}")
        flips[(z0_flip, y1_flip, z2_flip)] = (a_R, loss)
    
    best_flips = min(flips, key=lambda x: flips.get(x)[1])
    return best_flips, flips[best_flips]

def updateGD_alignment(flips, ap, a_R):
    ap_new = np.zeros(ap.shape)
    for i, a in enumerate(ap):
        ap_new[i] = [flips[0]*a[0], flips[1]*a[1], flips[2]*a[2]]

    angles_predicted = tf.convert_to_tensor(ap_new)

    angles_predicted_new = tf.math.add(angles_predicted, a_R)
    return angles_predicted_new

def training_angle_alignment_GD(steps, batch_size, projection_idx, learning_rate, angles_true, angles_predicted, optimization=True):
    optimizer = Adam(learning_rate=learning_rate)

    losses = np.empty(steps)
    time_start = time()

    euler = np.zeros(3, dtype=np.float64)
    a_R = [tf.Variable(euler)]

    for step in range(1, steps+1):

        # Sample some pairs.
        idx = list(np.random.choice(projection_idx, size=batch_size))
        
        # Compute distances between projections
        a_true = [angles_true[i] for i in idx]
        a_predicted = [angles_predicted[i] for i in idx]
        
        
        # Optimize by gradient descent.
        if optimization:
            losses[step-1], gradients = gradientGD(a_R, a_predicted, a_true)
            optimizer.apply_gradients(zip(gradients, a_R))
        else:
            losses[step-1] = lossGD(a_R, a_predicted, a_true)

        # Periodically report progress.
        if ((step % (steps//10)) == 0) or (step == steps):
            time_elapsed = time() - time_start
            print(f'step {step}/{steps} ({time_elapsed:.0f}s): loss = {losses[step-1]:.2e}')

    if optimization:
        # Plot convergence.
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, time()-time_start, steps), losses)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('loss');
    else:
        print(f"Mean distance difference: {np.mean(losses)}")

    return a_R, losses[-1]