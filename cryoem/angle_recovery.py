import time
import numpy as np
from IPython import display as IPyDisplay
from tensorflow.keras.optimizers import Adam
import seaborn as sns; sns.set(style="white", color_codes=True)
from tensorflow_graphics.geometry.transformation import quaternion
from cryoem.conversions import euler2quaternion, d_q, quaternion2euler
from cryoem.knn import get_knn_projections
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path


def angles_transpose(angles):
    """Transpose the angles"""
    angles = angles.copy()
    cols = [2, 1, 0]
    idx = np.empty_like(cols)
    idx[cols] = np.arange(len(cols))
    angles[:] = -angles[:, idx]
    return angles

def quaternion_constraint(low_ang, high_ang):
    """Constrain the quaternion unit vector amplitudes for the angle recovery"""
    def _inner(q):
        e = quaternion2euler(q)

        a0, a1, a2 = tf.unstack(e, axis=-1)
        a0 = tf.clip_by_value(a0, low_ang[0], high_ang[0])
        a1 = tf.clip_by_value(a1, low_ang[1], high_ang[1])
        a2 = tf.clip_by_value(a2, low_ang[2], high_ang[2])

        e_new = tf.stack((a0, a1, a2), axis=-1)
        q_new = euler2quaternion(e_new)
        return q_new

    return _inner


def train_angle_recovery(steps, batch_size, in_data, distance_fn, file_name, limit_distance=np.pi,
                         low_ang_const=[0.0, 0.0, 0.0], high_ang_const=[2.0, 0.4, 2.0], q_predicted=None,
                         angles_true=None, learning_rate=0.01, constraint=False): 
    """Main method for angle recovery.
    
    Parameters
    ----------
    steps : int
        Number of steps to run the angle recovery.
    batch_size : int
        The number of training data points utilized in one iteration/step. 
    in_data : np.array
        Array of projections or array of angles. Array of projections is used for the full pipeline
        when we have learned the distance between the projections. The array of angles is used when
        we want to perform angle recovery assuming we know the distance (and take the quaternion distance).
    distance_fn : callable
        The distance function. If we have the array of projections, this is the distance between two projections.
        If we have the array of angles, this is the distance between two quaternions (since angles will be converted to
        quaternions interally).
    file_name : str
        File name where to store the losses and the predicted orientation for every projection.
    limit_distance : float
        Default value: np.pi, used to limit the distance between two projections, predicting only the closer ones. 
        Not used anymore. 
    low_ang_const : list
        The lower bound for the angles we want to predict. Default value: [0.0, 0.0, 0.0], not used anymore.
    high_ang_const :
        The upper bound for the angles we want to predict. Default value: [2.0, 0.4, 2.0], not used anymore.
    q_predicted : np.ndarray, tf.Tensor
        Default value: None
    angles_true : np.array
        Default value: None
    learning_rate : float
        Optimizer's learning rate value. Default value: 0.01
    constraint : bool
        Whether to use the queaternion contraints or not. Default value: False

    Returns
    -------
    q_predicted : np.array
        Predicted orientations (in quaternion).
    losses : np.array
        Angle recovery losses.
    collect_daa : np.array
        Predicted orientations from every step of optimization.
    """                    

    time_start = time.time()
    collect_data = []
    optimizer = Adam(learning_rate=learning_rate)

    # low_ang = [0.0*np.pi, 0.0*np.pi, 0.0*np.pi]
    low_ang = list(map(lambda x: x*np.pi, low_ang_const))
    # high_ang = [2.0*np.pi, 0.4*np.pi, 2.0*np.pi] 
    high_ang = list(map(lambda x: x*np.pi, high_ang_const))           
    euler = np.random.uniform(low=[low_ang[0], low_ang[1], low_ang[2]], 
                          high=[high_ang[0], high_ang[1], high_ang[2]],
                          size=(len(in_data), 3))
    if q_predicted:
        # continue where left off
        if constraint:
            q_predicted = [tf.Variable(q, constraint=quaternion_constraint(low_ang, high_ang)) for q in q_predicted]
        else:
            q_predicted = [tf.Variable(q) for q in q_predicted]
    else:
        # optimize from scratch
        if constraint:
            q_predicted = [tf.Variable(q, constraint=quaternion_constraint(low_ang, high_ang)) for q in euler2quaternion(euler)]
        else:
            q_predicted = [tf.Variable(q) for q in euler2quaternion(euler)]

    if in_data.shape[1] == 3:
        in_data = euler2quaternion(in_data)

    losses = np.empty(steps)
    report = f"Shape of projections: {in_data.shape}"
    found_minimizer = False

    print(time.time()-time_start)

    for step, idx1, idx2 in sample_iter(steps, range(len(in_data)), batch_size, style="random"):
        q1 = [q_predicted[i] for i in idx1]
        q2 = [q_predicted[i] for i in idx2]
        q1 = np.array(q1)
        q2 = np.array(q2)

        # Compute distances
        in1 = [in_data[i] for i in idx1]
        in2 = [in_data[i] for i in idx2]
        in1 = np.array(in1)
        in2 = np.array(in2)

        distance_target = distance_fn(in1, in2)

        # WORK ONLY WITH LIMITED DISTANCES 
        indices_le2 = np.where(distance_target<limit_distance)[0]
        distance_target = np.take(distance_target, indices_le2)
        q1 = list(np.take(q1, indices_le2))
        q2 = list(np.take(q2, indices_le2))
        in1 = list(np.take(in1, indices_le2))
        in2 = list(np.take(in2, indices_le2))

        # Optimize by gradient descent.
        losses[step-1], gradients = gradient(q1, q2, distance_target)
        optimizer.apply_gradients(zip(gradients, q1 + q2))

        # Visualize progress periodically
        if step % 10 == 0:
            a = np.zeros((len(q_predicted), 4))
            for i, e in enumerate(q_predicted):
                a[i] = e.numpy()
            collect_data.append(a)

            plt.close();
            sns.set(style="white", color_codes=True)
            sns.set(style="whitegrid")

            if angles_true is not None:
                fig, axs = plt.subplots(1, 3, figsize=(24,7))

                # Optimization loss subplot
                axs[0].plot(np.linspace(0, time.time()-time_start, step), losses[:step], marker="o", lw=1, markersize=3)
                axs[0].set_xlabel('time [s]')
                axs[0].set_ylabel('loss');
                axs[0].set_title(f"[{step}/{steps}] Angle alignment optimization \nLOSS={np.mean(losses[step-10:step]):.2e} LR={learning_rate:.2e}")

                # NT - Distance count subplot (full)
                d2 = d_q(R.from_euler('zyz', angles_true).as_quat(), q_predicted)
                axs[1].set_xlim(0, np.pi)
                axs[1].set_title(f"[{step}/{steps}] Distances between true and predicted angles\nMEAN={np.mean(d2):.2e} rad ({np.degrees(np.mean(d2)):.2e}) STD={np.std(d2):.2e}")
                s = sns.distplot(d2, kde=False, bins=100, ax=axs[1], axlabel="Distance [rad]", color="r")
                max_count = int(max([h.get_height() for h in s.patches]))
                axs[1].plot([np.mean(d2)]*max_count, np.arange(0, max_count,1), c="r", lw=4)

                # T - Distance count subplot (full)
                angles_true_T = angles_transpose(angles_true)
                d2 = d_q(R.from_euler('zyz', angles_true_T).as_quat(), q_predicted)
                axs[2].set_xlim(0, np.pi)
                axs[2].set_title(f"[{step}/{steps}] TRANSPOSED Distances between true and predicted angles\nMEAN={np.mean(d2):.2e} rad ({np.degrees(np.mean(d2)):.2e}) STD={np.std(d2):.2e}")
                s = sns.distplot(d2, kde=False, bins=100, ax=axs[2], axlabel="Distance [rad]", color="r")
                max_count = int(max([h.get_height() for h in s.patches]))
                axs[2].plot([np.mean(d2)]*max_count, np.arange(0, max_count,1), c="r", lw=4)
            else:
                fig, axs = plt.subplots(figsize=(10,7))

                # Optimization loss subplot
                axs.plot(np.linspace(0, time.time()-time_start, step), losses[:step], marker="o", lw=1, markersize=3)
                axs.set_xlabel('time [s]')
                axs.set_ylabel('loss');
                axs.set_title(f"[{step}/{steps}] Angle recovery optimization \nLOSS={np.mean(losses[step-10:step]):.2e} LR={learning_rate:.2e}")


            IPyDisplay.clear_output(wait=True)
            IPyDisplay.display(plt.gcf())
            plt.close();
            time.sleep(0.1)

            Path(file_name).mkdir(parents=True, exist_ok=True)
            np.savez(file_name, quaternion.normalize(q_predicted).numpy(), losses, np.array(collect_data))

            if found_minimizer:
                time_elapsed = time.time() - time_start
                report += f'step {step}/{steps} ({time_elapsed:.0f}s): loss = {losses[step-1]:.2e}\n'
                break;

        # Periodically report progress.
        if ((step % (steps//10)) == 0) or (step == steps):
            time_elapsed = time.time() - time_start
            report += f'step {step}/{steps} ({time_elapsed:.0f}s): loss = {losses[step-1]:.2e}\n'

        if step >= 1001 and np.mean(losses[step-1001:step-1]) < 1e-8:
            found_minimizer = True

        if step >= 2001 and np.abs(np.mean(losses[step-1000:step-1])-np.mean(losses[step-2000:step-1000])) < 1e-7:
            found_minimizer = True

    print(report)
    return quaternion.normalize(q_predicted).numpy(), losses, np.array(collect_data)


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
            # this option is not used anymore, left for reference
            # select random sample for the first element of pair
            idx1 = list(np.random.choice(projection_idx, size=num_pairs))

            # half from kNN
            indices_p, distances_p, A_p = get_knn_projections(k=k)
            num_projections = len(projection_idx)
            idx2_knn = [indices_p[i][np.random.randint(1, k)] for i in idx1[:num_pairs//2]]
            idx2_random = list(np.random.randint(0, num_projections, num_pairs//2))
            # half random
            idx2 = idx2_knn + idx2_random

        yield step, idx1, idx2

def loss(q1_predicted, q2_predicted, distance_target):
    # The mean doesn't depend on the batch size.
    return tf.reduce_mean(tf.pow((d_q(q1_predicted, q2_predicted) - distance_target), 2))

def gradient(q1_predicted, q2_predicted, distance_target):
    with tf.GradientTape() as tape:
        loss_value = loss(q1_predicted, q2_predicted, distance_target)
    gradient = tape.gradient(loss_value, q1_predicted + q2_predicted)

    return loss_value, gradient

