from cryoem.quaternions import euler2quaternion, d_q
from cryoem.projections import RotationMatrix
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow_graphics.geometry.transformation import quaternion
from time import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white", color_codes=True)
from itertools import combinations 
import random
from tensorflow_graphics.math import vector
from scipy.interpolate import interp1d
from cryoem.knn import get_knn_projections



def sample_iter(steps, projection_idx, num_pairs, style="random", k=None):

    for step in range(1, steps+1):
        if not k and style != "random":
            raise ValueError("Please specify k for kNN for sample_pairs method")
        
        if style=="random":
            idx1 = list(np.random.choice(projection_idx, size=num_pairs))
            idx2 = list(np.random.choice(projection_idx, size=num_pairs))
        
        elif style=="knn":
            idx1 = list(np.random.choice(projection_idx, size=num_pairs))
            indices_p, distances_p, A_p = get_knn_projections(k=k)
            idx2 = [indices_p[i][np.random.randint(1, k)] for i in idx1]
    
        elif style=="knn_and_random":
            # select random sample for the first element of pair
            idx1 = list(np.random.choice(projection_idx, size=num_pairs))
            
            # half from kNN
            indices_p, distances_p, A_p = get_knn_projections(k=k)
            idx2_knn = [indices_p[i][np.random.randint(1, k)] for i in idx1[:num_pairs//2]]
            idx2_random = list(np.random.randint(0, num_projections, num_pairs//2))
            # half random
            idx2 = idx2_knn + idx2_random
        
        yield step, idx1, idx2

def get_projection(i, j):
    u = np.array([i, j]) # vector u 
    v = np.array([1, 1]) # vector v: 

    v_norm = np.sqrt(sum(v**2))

    proj_of_u_on_v = (np.dot(u, v)/v_norm**2)*v 

    return proj_of_u_on_v[0] #, proj_of_u_on_v[1]

# def mod_angles(angles):
#     for i, a in enumerate(angles):
#         angles[i] = np.array([ a[0]%(2*np.pi), a[1]%(np.pi), a[2]%(2*np.pi)])
#     return angles

def loss(a1_predicted, a2_predicted, distance_target, dt_type, space):
    a1_predicted = list(a1_predicted)
    a2_predicted = list(a2_predicted)
    
    q1 = euler2quaternion(a1_predicted)
    q2 = euler2quaternion(a2_predicted)
    
    distance = d_q(q1, q2)
    
    if space == "dQspace":
        if dt_type == "dP":
            # Convert dP to dQ
            # TODO: fix:
            intercept = 0.0
            slope = 1.0
            distance_target = (distance_target-intercept)/slope
        # elif dt_type == "dQ": ##################
        #     distance_target = np.array([ get_projection(i, j) for i, j in zip(distance.numpy(), distance_target.T[0])])
        #     distance_target = distance_target.T
    elif space == "dPspace":
        if dt_type == "dP":
            # Convert dQ to dP
            distance = tf.math.polyval(coeffs4dP, distance)
        elif dt_type == "dQ":
            distance = tf.math.polyval(coeffs4dP, distance)
            distance_target = tf.math.polyval(coeffs4dP, distance_target)  

    # The mean doesn't depend on the batch size.
    return tf.reduce_mean((distance - distance_target)**2)

def gradient(a1_predicted, a2_predicted, distance_target, dt_type, space):
    with tf.GradientTape() as tape:
        loss_value = loss(a1_predicted, a2_predicted, distance_target, dt_type, space)
        gradient = tape.gradient(loss_value, a1_predicted + a2_predicted)
        
    return loss_value, gradient

# def loss_global_distance_difference( a_predicted, a_true):
#     a_predicted = list(a_predicted)
#     a_true = list(a_true)

#     q_predicted = euler2quaternion(a_predicted)
#     q_true = euler2quaternion(a_true)
    
#     distance = d_q(q_true, q_predicted)
    
#     return tf.reduce_mean(distance)


# def loss_predicted_vs_true_angle(steps, batch_size, projection_idx, learning_rate=0.01, optimization=False, angles_predicted=None, angles_true=None):

#     losses = np.empty(steps)
#     time_start = time()
#     optimizer = Adam(learning_rate=learning_rate)
    
#     for step, idx1, idx2 in sample_iter(projection_idx, batch_size, style="random"):

#         a1 = [angles_predicted[i] for i in idx1]
#         a2 = [angles_predicted[i] for i in idx2]

#         # Compute distances between true quaternions
#         a1_true = [angles_true[i] for i in idx1]
#         a2_true = [angles_true[i] for i in idx2]
#         q1_true = euler2quaternion(a1_true)
#         q2_true = euler2quaternion(a2_true)
        
#         distance_target = d_q(q1_true, q2_true)

#         # Optimize by gradient descent.
#         if optimization:
#             losses[step-1], gradients = gradient(a1, a2, distance_target, dt_type="dQ", space="dQspace")
#             optimizer.apply_gradients(zip(gradients, a1 + a2))
#         else:
#             losses[step-1] = loss(a1, a2, distance_target, dt_type="dQ", space="dQspace")
        
#         # Periodically report progress.
#         if ((step % (steps//10)) == 0) or (step == steps):
#             time_elapsed = time() - time_start
#             #loss_mean = np.mean(losses[(step-1)-(steps//10):step-1])
#             print(f'step {step}/{steps} ({time_elapsed:.0f}s): loss = {losses[step-1]:.2e}')

#     if optimization:
#         # Plot convergence.
#         sns.set(style="white", color_codes=True)
#         sns.set(style="whitegrid")
#         fig, ax = plt.subplots(figsize=(15,10))
#         sns.lineplot(x=np.linspace(0, time()-time_start, steps), y=losses, ax=ax, marker="o")
#         ax.set(xlabel='time [s]', ylabel='loss')
#         plt.show()
#     else:
#         print(f"Mean loss: {np.mean(losses)}")



# def loss_predicted_angle_vs_projection(steps, batch_size, optimization=False, angles_predicted=None):
#     losses = np.empty(steps)
#     time_start = time()
#     optimizer = Adam(learning_rate=0.001)
    
#     for step, idx1, idx2 in sample_iter(projection_idx, batch_size, style="random"):

#         a1 = [angles_predicted[i] for i in idx1]
#         a2 = [angles_predicted[i] for i in idx2]

#         # Compute distances between projections
#         p1 = [X[i] for i in idx1]
#         p2 = [X[i] for i in idx2]

#         distance_target = d_p(p1, p2)


#         # manual checkpoint
#         angles_predicted_final = np.zeros((NUM_PROJECTIONS, 3))
#         for i, ap in enumerate(angles_predicted):
#             angles_predicted_final[i] = ap.numpy()

#         np.save('data/angles_predicted_final.npy', angles_predicted_final)

#         # Optimize by gradient descent.
#         if optimization:
#             losses[step-1], gradients = gradient(a1, a2, distance_target, dt_type="dP", space="dQspace")
#             optimizer.apply_gradients(zip(gradients, a1 + a2))
#         else:
#             losses[step-1] = loss(a1, a2, distance_target, dt_type="dP", space="dQspace")

#         # Periodically report progress.
#         if ((step % (steps//10)) == 0) or (step == steps):
#             time_elapsed = time() - time_start
#             print(f'step {step}/{steps} ({time_elapsed:.0f}s): loss = {losses[step-1]:.2e}')


#     if optimization:
#         # Plot convergence.
#         fig, ax = plt.subplots()
#         ax.plot(np.linspace(0, time()-time_start, steps), losses)
#         ax.set_xlabel('time [s]')
#         ax.set_ylabel('loss');
#     else:
#         print(f"Mean loss: {np.mean(losses)}")



def train_angle_recovery(steps, batch_size, projection_idx, 
                        angles_predicted, 
                        est_dist_input, est_dist, 
                        learning_rate=0.01, 
                        optimization=False):
                        
    if est_dist_input.shape[1] == 3:
        convert = euler2quaternion
        dt_type = "dQ"
    else:
        convert = lambda x: x 
        dt_type = "dP"
    
    losses = np.empty(steps)
    losses_avg_dist_diff = np.empty(steps)
    time_start = time()
    optimizer = Adam(learning_rate=learning_rate)
    
    for step, idx1, idx2 in sample_iter(steps, projection_idx, batch_size, style="random"):

        a1 = [angles_predicted[i] for i in idx1]
        a2 = [angles_predicted[i] for i in idx2]

        # Compute distances
        in1 = convert([est_dist_input[i] for i in idx1])
        in2 = convert([est_dist_input[i] for i in idx2])
        
        distance_target = est_dist(in1, in2)

        # Optimize by gradient descent.
        if optimization:
            losses[step-1], gradients = gradient(a1, a2, distance_target, dt_type=dt_type, space="dQspace")
            optimizer.apply_gradients(zip(gradients, a1 + a2))
            # if dt_type == "dQ":
            #     losses_avg_dist_diff[step-1], _ = distance_difference(angles_predicted, est_dist_input)
        else:
            losses[step-1] = loss(a1, a2, distance_target, dt_type=dt_type, space="dQspace")
        
        # Periodically report progress.
        if ((step % (steps//10)) == 0) or (step == steps):
            time_elapsed = time() - time_start
            print(f'step {step}/{steps} ({time_elapsed:.0f}s): loss = {losses[step-1]:.2e}')

            # if dt_type == "dQ":
            #     print(f'\tavg. distance difference [degree]: {np.degrees(losses_avg_dist_diff[step-1]):.2f}')
    if optimization:
        # Plot convergence.
        sns.set(style="white", color_codes=True)
        sns.set(style="whitegrid")
        if dt_type == "dQ":
            fig, ax = plt.subplots(1, 1, figsize=(15,10))
            sns.lineplot(x=np.linspace(0, time()-time_start, steps), y=losses, ax=ax, marker="o")
            #sns.lineplot(x=np.linspace(0, time()-time_start, steps), y=losses_avg_dist_diff, ax=ax[1], marker="o", color="red")   
            ax.set(xlabel='time [s]', ylabel='loss')
            #ax[1].set(xlabel='time [s]', ylabel='avg distance difference')
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(15,15))
            sns.lineplot(x=np.linspace(0, time()-time_start, steps), y=losses, ax=ax, marker="o")
            ax.set(xlabel='time [s]', ylabel='loss')
            plt.show()
    else:
        print(f"Mean loss: {np.mean(losses)}")

### Error checking with alignment method
def euclidean_distance(a,b):
    return tf.reduce_mean(tf.norm(a-b, ord='euclidean', axis=1))

def geodesic_distance(a, b):
    return tf.reduce_mean(tf.acos(vector.dot(a, b, keepdims=False)))
    

def collect_points_z3(step, angles, angles_true, distance_type):
    _ap = np.zeros(angles.shape)
    for j, a in enumerate(angles):
        _ap[j] = [a[0]%(2*np.pi), a[1]%(2*np.pi), (a[2]+step)%(2*np.pi)]

    ats = RotationMatrix(angles_true)[:,:3]
    aps = RotationMatrix(_ap)[:,:3]

    loss = distance_type(ats, aps).numpy()
        
    return step, loss

def symmetric_z3(angles):
    _ap = np.zeros(angles.shape)
    for j, a in enumerate(angles):
        _ap[j] = [a[0]%(2*np.pi), a[1]%(2*np.pi), (-a[2])%(2*np.pi)]
        
    return _ap

def symmetric_y2(angles):
    _ap = np.zeros(angles.shape)
    for j, a in enumerate(angles):
        _ap[j] = [a[0]%(2*np.pi), (-a[1])%(2*np.pi), a[2]%(2*np.pi)]
        
    return _ap


def collect_points_y2(step, angles, angles_true, distance_type):
    _ap = np.zeros(angles.shape)
    for j, a in enumerate(angles):
        _ap[j] = [a[0]%(2*np.pi), (a[1]+step)%(2*np.pi), (a[2])%(2*np.pi)]

    ats = RotationMatrix(angles_true)[:,:3]
    aps = RotationMatrix(_ap)[:,:3]

    loss = distance_type(ats, aps).numpy()
        
    return step, loss

def collect_points_z1(step, angles, angles_true, distance_type):
    _ap = np.zeros(angles.shape)
    for j, a in enumerate(angles):
        _ap[j] = [(a[0]+step)%(2*np.pi), (a[1])%(2*np.pi), (a[2])%(2*np.pi)]

    ats = RotationMatrix(angles_true)[:,:3]
    aps = RotationMatrix(_ap)[:,:3]

    loss = distance_type(ats, aps).numpy()
        
    return step, loss


def find_best_rotation_about_axis(axis_fn, distance_type, steps, angles, angles_true):
    x = np.zeros(len(steps))
    y = np.zeros(len(steps))
    for i, step in enumerate(steps):
        step, loss = axis_fn(step, angles, angles_true, distance_type)
        x[i] = step
        y[i] = loss
    
    interpolation = interp1d(x, y, kind='cubic')
    step_new = np.linspace(min(steps), max(steps), num=100, endpoint=True)
    loss_new = interpolation(step_new)
    
    min_idx = np.argmin(loss_new)
    min_loss = loss_new[min_idx]
    min_step = step_new[min_idx]
    
    plt.plot(x, y, 'o', step_new, loss_new, '--')
    plt.legend(['data', 'cubic'], loc='best')
    plt.show()
    
    return min_step, min_loss 

def find_best_rotation(angles, angles_true, steps, distance_type=euclidean_distance):
    ### default
    best_z3_step, min_z3_loss = find_best_rotation_about_axis(collect_points_z3, distance_type, steps, angles, angles_true)
    print("step on z3 axis: ", best_z3_step, " loss: ", min_z3_loss, " rad (", np.degrees(min_z3_loss), " degrees)")
    
    # implement change on z3 axis
    angles_updated_z3 = np.zeros(angles.shape)
    for j, a in enumerate(angles):
        angles_updated_z3[j] = [(a[0])%(2*np.pi), (a[1])%(2*np.pi), (a[2]+best_z3_step)%(2*np.pi)]
    
    best_y2_step, min_y2_loss = find_best_rotation_about_axis(collect_points_y2, distance_type, steps, angles_updated_z3, angles_true)
    print("step on y2 axis: ", best_y2_step, " loss: ", min_y2_loss, " rad (", np.degrees(min_y2_loss), " degrees)")
    
    ### symmetric z3
    angles = symmetric_z3(angles)
    
    symz3_best_z3_step, symz3_min_z3_loss = find_best_rotation_about_axis(collect_points_z3, distance_type, steps, angles, angles_true)
    print("symmetric step on z3 axis: ", symz3_best_z3_step, " loss: ", symz3_min_z3_loss, " rad (", np.degrees(symz3_min_z3_loss), " degrees)")
    
    # implement change on z3 axis
    symz3_angles_updated_z3 = np.zeros(angles.shape)
    for j, a in enumerate(angles):
        symz3_angles_updated_z3[j] = [(a[0])%(2*np.pi), (a[1])%(2*np.pi), (a[2]+best_z3_step)%(2*np.pi)]
    
    symz3_best_y2_step, symz3_min_y2_loss = find_best_rotation_about_axis(collect_points_y2, distance_type, steps, symz3_angles_updated_z3, angles_true)
    print("symmetric step on y2 axis: ", symz3_best_y2_step, " loss: ", symz3_min_y2_loss, " rad (", np.degrees(symz3_min_y2_loss), " degrees)")
    
    ### symmetric y2
    angles = symmetric_y2(angles)
    
    symy2_best_z3_step, symy2_min_z3_loss = find_best_rotation_about_axis(collect_points_z3, distance_type, steps, angles, angles_true)
    print("symmetric step on z3 axis: ", symy2_best_z3_step, " loss: ", symy2_min_z3_loss, " rad (", np.degrees(symy2_min_z3_loss), " degrees)")
    
    # implement change on z3 axis
    symy2_angles_updated_z3 = np.zeros(angles.shape)
    for j, a in enumerate(angles):
        symy2_angles_updated_z3[j] = [(a[0])%(2*np.pi), (a[1])%(2*np.pi), (a[2]+best_z3_step)%(2*np.pi)]
    
    symy2_best_y2_step, symy2_min_y2_loss = find_best_rotation_about_axis(collect_points_y2, distance_type, steps, symy2_angles_updated_z3, angles_true)
    print("symmetric step on y2 axis: ", symy2_best_y2_step, " loss: ", symy2_min_y2_loss, " rad (", np.degrees(symy2_min_y2_loss), " degrees)")
    
    idx = np.argmin(np.array([min_y2_loss, symz3_min_y2_loss, symy2_min_y2_loss]))
    symmetries = ["no symmetry", "symmetric Z3", "symmetric Y2"]
    step_z3 = [best_z3_step, symz3_best_z3_step, symy2_best_z3_step]
    step_y2 = [best_y2_step, symz3_best_y2_step, symy2_best_y2_step]

    return symmetries[idx], step_z3[idx], step_y2[idx]

    
def update_angles(ap, symmetric, z3_rotation, y2_rotation):
    sign_z3 = -1 if symmetric=="symmetric Z3" else 1
    sign_y2 = -1 if symmetric=="symmetric Y2" else 1
    
    _ap = np.zeros(ap.shape)
    for j, a in enumerate(ap):
        _ap[j] = [a[0]%(2*np.pi), (sign_y2*a[1]+y2_rotation)%(2*np.pi), (sign_z3*a[2]+z3_rotation)%(2*np.pi)]

    return _ap

def distance_difference(angles_predicted, angles_true):
    aps = RotationMatrix(angles_predicted)[:,:3]
    ats = RotationMatrix(angles_true)[:,:3]

    ed = euclidean_distance(ats, aps).numpy()
    gd = geodesic_distance(ats, aps).numpy()
    print("Euclidean distance: ", ed)
    print("Geodesic distance: ", gd, " rad (", np.degrees(gd), " degrees)")

    q_predicted = euler2quaternion(angles_predicted)
    q_true = euler2quaternion(angles_true)
    qd = np.mean(d_q(q_predicted, q_true).numpy())
    print("Quaternion distance: ", qd)

    return ed, gd, qd





##### With Epochs #######

def data_iter(batch_size, projection_ids):
    projection_idx = list(combinations(projection_ids, 2)) 
    
    idx = list(range(len(projection_idx)))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0, len(projection_idx), batch_size)):
        j = np.array(idx[i: min(i + batch_size, len(projection_idx))])

        projection_pairs = np.take(projection_idx, j, axis=0)
        p1s = projection_pairs[:, 0]
        p2s = projection_pairs[:, 1]

        yield batch_i, p1s, p2s

def loss_predicted_vs_true_angle_with_epochs(epochs, batch_size, projection_ids, learning_rate=0.01, optimization=False, angles_predicted=None, angles_true=None):
    """Only one step takes 38700s for 5K projecections, and there is 12205 in epoch -> 14 years for 1 epochs
    loss_predicted_vs_true_angle(epochs=3, 
                             batch_size=1024, 
                             projection_ids=range(NUM_PROJECTIONS), 
                             learning_rate=0.01, 
                             optimization=True, 
                             angles_predicted=angles_predicted, 
                             angles_true=angles_true)
    """
    epoch_losses = np.empty(epochs)
    time_start = time()
    optimizer = Adam(learning_rate=learning_rate)
    
    for epoch in range(1, epochs+1):
        losses = []
        # Decay learning rate.
        if epoch > 2:
            learning_rate *= 0.1
            
        # Sample some pairs.
        #idx1, idx2 = sample_pairs(projection_idx=projection_idx, num_pairs=batch_size, style="random")
        steps = len(list(data_iter(batch_size, projection_ids)))
        for batch_i, idx1, idx2 in data_iter(batch_size, projection_ids):
            a1 = [angles_predicted[i] for i in idx1]
            a2 = [angles_predicted[i] for i in idx2]

            # Compute distances between true quaternions
            a1_true = [angles_true[i] for i in idx1]
            a2_true = [angles_true[i] for i in idx2]
            q1_true = euler2quaternion(a1_true)
            q2_true = euler2quaternion(a2_true)

            distance_target = d_q(q1_true, q2_true)

            # Optimize by gradient descent.
            if optimization:
                loss_value, gradients = gradient(a1, a2, distance_target, dt_type="dQ", space="dQspace")

                optimizer.apply_gradients(zip(gradients, a1 + a2))
            else:
                loss_value = loss(a1, a2, distance_target, dt_type="dQ", space="dQspace")
            losses.append(loss_value)

            # Periodically report progress.
            if ((batch_i % (steps//10)) == 0) or (batch_i == steps):
                time_elapsed = time() - time_start
                #loss_mean = np.mean(losses[(step-1)-(steps//10):step-1])
                print(f'\tstep {batch_i}/{steps} ({time_elapsed:.0f}s): loss = {loss_value:.2e}')

        
        epoch_losses[epoch-1] = np.median(losses)
        
    # Periodically report progress.
    print(f'epoch {epoch}/{epochs} ({time_elapsed:.0f}s): loss = {epoch_losses[epoch-1]:.2e}')

    if optimization:
        sns.set(style="white", color_codes=True)
        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(15,10))
        sns.lineplot(x=np.linspace(0, time()-time_start, epochs), y=epoch_losses, ax=ax, marker="o")
        ax.set(xlabel='time [s]', ylabel='loss')
        plt.show()
    else:
        print(f"Mean loss: {np.mean(epoch_losses)}")