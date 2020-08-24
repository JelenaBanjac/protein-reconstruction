import matplotlib.pyplot as plt
import ipyvolume as ipv
from cryoem.rotation_matrices import RotationMatrix
from scipy.spatial.transform import Rotation as R
import numpy as np
import tensorflow as tf
import seaborn as sns; sns.set(style="white", color_codes=True)
import pandas as pd
from matplotlib._png import read_png
from cryoem.conversions import euler2quaternion, d_q

fg_color = 'white'
bg_color = 'black'

def _plot(image, title, ax, colorbar=False):
    # NOTE: used
    im = ax.imshow(image)
  
    # set title plus title color
    ax.set_title(title, color=fg_color)

    # set figure facecolor
    ax.patch.set_facecolor(bg_color)

    # set tick and ticklabel color
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    for spine in im.axes.spines.values():
        spine.set_edgecolor(fg_color)    

    if colorbar:
        cb = plt.colorbar(im)
        # set colorbar label plus label color
        cb.set_label('Pixel value', color=fg_color)

        # set colorbar tick color
        cb.ax.yaxis.set_tick_params(color=fg_color)

        # set colorbar edgecolor 
        cb.outline.set_edgecolor(fg_color)

        # set colorbar ticklabels
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)
        
def plot_projection(projection, angles):
    # NOTE: used
    projection = np.squeeze(projection)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    _plot(projection, f"Angles: {', '.join([f'{x:.2f}' for x in angles])} [rad]", ax, colorbar=True)

    fig.patch.set_facecolor(bg_color) 
    plt.tight_layout()
    plt.show()
    
def plot_projections(images, titles, nrows=2, ncols=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 10))
    
    cr = [(i, j) for i in range(nrows) for j in range(ncols)]
    
    for image, title, (i, j) in zip(images, titles, cr):
        
        _plot(image, title, axes[i][j] if nrows>1 else axes[j], colorbar=False)

    fig.patch.set_facecolor(bg_color)    
    plt.tight_layout()

    plt.show()

##################### SPHERES #####################

def save_space_plot(filename):
    ipv.save(f"{filename}.html")
    ipv.savefig(f"{filename}.png")


def plot_detector_pixels(angles):
    # NOTE: used
    ipv.clear()

    if isinstance(angles[0], tf.Variable):
        angles = np.array([a.numpy() for a in angles])

    arr = RotationMatrix(angles)

    ipv.figure(width=500, height=500)
    ipv.scatter(arr[:,0], arr[:,1], arr[:,2], marker="sphere", color="blue", size=1)
    ipv.xlim(-1,1);ipv.ylim(-1,1);ipv.zlim(-1,1)
    ipv.show()

def plot_one_closest_vs_all_in_euclidean_space(angles, closest):
    ipv.clear()
    knn = RotationMatrix(np.take(angles, closest, axis=0))
    all = RotationMatrix(np.delete(angles, closest, 0))

    ipv.figure(width=500, height=400)
    ipv.scatter(knn[:,0], knn[:,1], knn[:,2], marker="diamond", color="red", size=1.5)
    ipv.scatter(all[:,0], all[:,2], all[:,1], marker="sphere", color="blue", size=1)
    ipv.show()

def plot_only_selected_in_euclidean_space(angles, angles_true, selected):
    ipv.clear()
    aps = RotationMatrix(np.take(angles, selected, axis=0))[:,:3]
    ats = RotationMatrix(np.take(angles_true, selected, axis=0))[:,:3]
    connections = np.empty((len(selected)*2, 2,3))
    connections[:] = np.nan
    connections[::2] = np.stack([ats, aps],axis=1)
    
    ipv.scatter(ats[:,0], ats[:,1], ats[:,2], marker="sphere", color="green", size=1)
    ipv.scatter(aps[:,0], aps[:,1], aps[:,2], marker="sphere", color="red", size=1)
    ipv.plot(connections[:,:,0].flatten(),connections[:,:,1].flatten(), connections[:,:,2].flatten(),color="red", lynestyle="--")
    ipv.xlim(-1, 1);ipv.ylim(-1,1);ipv.zlim(-1, 1)
    ipv.show()
 

def plot_only_closest_in_euclidean_space(angles, closest):
    ipv.clear()
    for i in closest:
        main = RotationMatrix(np.array([angles[i[0]]]))
        knn = RotationMatrix(np.take(angles, i[1:], axis=0))
        
        ipv.scatter(main[:,0], main[:,1], main[:,2], marker="diamond", color="red", size=1.5)
        ipv.scatter(knn[:,0], knn[:,1], knn[:,2], marker="sphere", color="blue", size=1)
        
        for c in range(5):
            connection0 = [main[:,0][0], knn[c,0]]
            connection1 = [main[:,1][0], knn[c,1]]
            connection2 = [main[:,2][0], knn[c,2]]
            ipv.plot(connection0, connection1, connection2,color="red", lynestyle="--")
        
    ipv.xlim(-1, 1);ipv.ylim(-1,1);ipv.zlim(-1, 1)
    ipv.show()
 
def plot_iterations_polar_plot(q_all, angles_true, interval=1, connections=True, selected=None):
    if q_all.shape[1] != angles_true.shape[0]:
        raise Exception("Should specify the same number of true angles and predicted angles")
    
    if not selected:
        selected = range(len(angles_true))
        
    ipv.clear()
    angles_predicted_all = np.zeros((*q_all.shape[0:2], 3))
    for i, q in enumerate(q_all):
        angles_predicted_all[i, :] = R.from_quat(q).as_euler("zyz")% (2*np.pi)

    # PREDICTED ANGLES ITERATION    
    xyz = np.zeros(angles_predicted_all.shape)
    for i, a in enumerate(angles_predicted_all):
        z0, y1, z1 = a[:,0], a[:,1], a[:,2]
        x = z1*np.sin(y1)*np.cos(z0)
        y = z1*np.sin(y1)*np.sin(z0)
        z = z1*np.cos(y1)
        xyz[i,:,:] = np.array([x, y, z]).T
        
    # TRUE ANGLES
    xyz_true = np.zeros(angles_true.shape)
    z0, y1, z1 = angles_true[:,0], angles_true[:,1], angles_true[:,2]

    x = z1*np.sin(y1)*np.cos(z0)
    y = z1*np.sin(y1)*np.sin(z0)
    z = z1*np.cos(y1)

    xyz_true = np.array([x, y, z]).T
    
    xyz_true = np.take(xyz_true, selected, axis=0)
    xyz = np.take(xyz, selected, axis=1)
    xyzt = np.stack((xyz_true,)*len(xyz), axis=0)
    
    N = len(selected)
    steps = len(xyz)
    x = np.zeros((steps, 2*N))
    y = np.zeros((steps, 2*N))
    z = np.zeros((steps, 2*N))
    
    for i in range(0, steps):
        x[i] = np.concatenate([xyz_true[:,0], xyz[i,:,0]])
        y[i] = np.concatenate([xyz_true[:,1], xyz[i,:,1]])
        z[i] = np.concatenate([xyz_true[:,2], xyz[i,:,2]])
    
    ipv.figure()
    lines = [[i, i+N] for i in range(N)]
        
    if connections:
        s = ipv.scatter(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], color="blue", marker="sphere")
        ipv.scatter(xyz_true[:,0], xyz_true[:,1], xyz_true[:,2], marker="sphere", color="red", size=2)
        p = ipv.plot_trisurf(x, y, z, lines=lines);
        ipv.animation_control([s, p], interval=interval)
    else:
        s = ipv.scatter(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], color="blue", marker="sphere")
        ipv.scatter(xyz_true[:,0], xyz_true[:,1], xyz_true[:,2], marker="sphere", color="red", size=2)
        ipv.animation_control(s, interval=interval)

    ipv.xlim(-2*np.pi, 2*np.pi);ipv.ylim(-2*np.pi, 2*np.pi);ipv.zlim(-2*np.pi, 2*np.pi);
    ipv.show()

def plot_iterations_rotvec(q_all, angles_true, interval=1, connections=True, selected=None):
    ipv.clear()
    if q_all.shape[1] != angles_true.shape[0]:
        raise Exception("Should specify the same number of true angles and predicted angles")

    if not selected:
        selected = range(len(angles_true))

    ipv.clear()
    angles_predicted_all = np.zeros((*q_all.shape[0:2], 3))
    for i, q in enumerate(q_all):
        angles_predicted_all[i, :] = R.from_quat(q).as_euler("zyz")% (2*np.pi)

    # PREDICTED ANGLES ITERATION    
    xyz = np.zeros(angles_predicted_all.shape)
    for i, a in enumerate(angles_predicted_all):
        #z0, y1, z1 = a[:,0], a[:,1], a[:,2]
        rv = R.from_euler('zyz', a).as_rotvec()
        x = rv[:,0]
        y = rv[:,1]
        z = rv[:,2]
        xyz[i,:,:] = np.array([x, y, z]).T

    # TRUE ANGLES
    xyz_true = np.zeros(angles_true.shape)
    #z0, y1, z1 = angles_true[:,0], angles_true[:,1], angles_true[:,2]
    rvt = R.from_euler('zyz', angles_true).as_rotvec()

    x = rvt[:,0]
    y = rvt[:,1]
    z = rvt[:,2]

    xyz_true = np.array([x, y, z]).T

    xyz_true = np.take(xyz_true, selected, axis=0)
    xyz = np.take(xyz, selected, axis=1)
    xyzt = np.stack((xyz_true,)*len(xyz), axis=0)

    N = len(selected)
    steps = len(xyz)
    x = np.zeros((steps, 2*N))
    y = np.zeros((steps, 2*N))
    z = np.zeros((steps, 2*N))

    for i in range(0, steps):
        x[i] = np.concatenate([xyz_true[:,0], xyz[i,:,0]])
        y[i] = np.concatenate([xyz_true[:,1], xyz[i,:,1]])
        z[i] = np.concatenate([xyz_true[:,2], xyz[i,:,2]])

    ipv.figure()
    lines = [[i, i+N] for i in range(N)]

    if connections:
        s = ipv.scatter(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], color="blue", marker="sphere")
        ipv.scatter(xyz_true[:,0], xyz_true[:,1], xyz_true[:,2], marker="sphere", color="red", size=2)
        p = ipv.plot_trisurf(x, y, z, lines=lines);
        ipv.animation_control([s, p], interval=interval)
    else:
        s = ipv.scatter(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], color="blue", marker="sphere")
        ipv.scatter(xyz_true[:,0], xyz_true[:,1], xyz_true[:,2], marker="sphere", color="red", size=2)
        ipv.animation_control(s, interval=interval)

    ipv.xlim(-np.pi, np.pi);ipv.ylim(-np.pi, np.pi);ipv.zlim(-np.pi, np.pi);
    ipv.show()

def plot_rotvec(angles_true):
    a = R.from_euler('zyz', angles_true).as_rotvec()

    ipv.figure()
    ipv.scatter(a[:,0], a[:,1], a[:,2], marker="sphere", color="red", size=1)
    ipv.xlim(-np.pi, np.pi);ipv.ylim(-np.pi, np.pi);ipv.zlim(-np.pi, np.pi);
    ipv.show()

def plot_polar_plot(angles_true):
    z0, y1, z1 = angles_true[:,0], angles_true[:,1], angles_true[:,2]

    ipv.figure()
    x = z1*np.sin(y1)*np.cos(z0)
    y = z1*np.sin(y1)*np.sin(z0)
    z = z1*np.cos(y1)

    ipv.scatter(x, y, z, marker="sphere", color="red", size=1)
    ipv.show()

# def plot_SO3_space(angles, rotation_axes="zyz", normalized=False):
#     """ zyz - intrinsic rotation ()
#         ZYZ - extrinsic rotation (rotates about axes of motionless coordinate system)
#     """
#     ipv.clear()

#     if isinstance(angles[0], tf.Variable):
#         angles = np.array([a.numpy() for a in angles])

#     rotation_vectors = R.from_euler(seq=rotation_axes, angles=angles).as_rotvec()
#     magnitude = np.linalg.norm(rotation_vectors, axis=1)

#     if normalized:
#         rotation_vectors /= magnitude[:, np.newaxis]
#         ipv.xlim(-1, 1);ipv.ylim(-1,1);ipv.zlim(-1, 1)
#     else:
#         ipv.xlim(-np.pi, np.pi);ipv.ylim(-np.pi,np.pi);ipv.zlim(-np.pi, np.pi)

#     ipv.figure(width=500, height=500)
#     ipv.scatter(rotation_vectors[:,0], rotation_vectors[:,1], rotation_vectors[:,2], marker="sphere", color="blue", size=1)
    
#     ipv.show()


# def plot_one_closest_vs_all_in_SO3_space(angles, closest):
#     ipv.clear()
#     rotation_vectors = R.from_euler(seq="zyz", angles=angles).as_rotvec()
#     magnitude = np.linalg.norm(rotation_vectors, axis=1)

#     knn = np.take(rotation_vectors, closest, axis=0)
#     all = np.delete(rotation_vectors, closest, 0)

#     ipv.figure(width=500, height=400)
#     ipv.scatter(knn[:,0], knn[:,1], knn[:,2], marker="diamond", color="red", size=1.5)
#     ipv.scatter(all[:,0], all[:,2], all[:,1], marker="sphere", color="blue", size=1)
#     ipv.show()

# def plot_only_closest_in_SO3_space(angles, closest):
#     ipv.clear()
#     rotation_vectors = R.from_euler(seq="zyz", angles=angles).as_rotvec()
#     magnitude = np.linalg.norm(rotation_vectors, axis=1)

#     for i in closest:
#         main = np.array([rotation_vectors[i[0]]])
#         knn = np.take(rotation_vectors, i[1:], axis=0)
        
#         ipv.scatter(main[:,0], main[:,1], main[:,2], marker="diamond", color="red", size=1.5)
#         ipv.scatter(knn[:,0], knn[:,1], knn[:,2], marker="sphere", color="blue", size=1)
        
#         for c in range(5):
#             connection0 = [main[:,0][0], knn[c,0]]
#             connection1 = [main[:,1][0], knn[c,1]]
#             connection2 = [main[:,2][0], knn[c,2]]
#             ipv.plot(connection0, connection1, connection2,color="red", lynestyle="--")
        
#     ipv.xlim(-np.pi, np.pi);ipv.ylim(-np.pi,np.pi);ipv.zlim(-np.pi, np.pi)
#     ipv.show()


def plot_rays(angles, indices):
    arr = RotationMatrix(angles)

    ipv.clear()
    ipv.figure(width=500, height=500)
    indices=indices if indices else range(len(arr))
    scale = 0.2

    for i in indices:
        ipv.scatter(arr[i:i+1,0], arr[i:i+1,1], arr[i:i+1,2], marker="sphere", color="blue", size=1)
        ipv.scatter(arr[i,0]+arr[i:i+1,6]*scale, arr[i,1]+arr[i:i+1,7]*scale, arr[i,2]+arr[i:i+1,8]*scale, marker="sphere", color="red", size=1)
        ipv.scatter(arr[i,0]+arr[i:i+1,9]*scale, arr[i,1]+arr[i:i+1,10]*scale, arr[i,2]+arr[i:i+1,11]*scale, marker="sphere", color="green", size=1)


        connection0 = [arr[i,0], arr[i,0]+arr[i,6]*scale]
        connection1 = [arr[i,1], arr[i,1]+arr[i,7]*scale]
        connection2 = [arr[i,2], arr[i,2]+arr[i,8]*scale]
        ipv.plot(connection0, connection1, connection2,color="red", lynestyle="--")
        connection0 = [arr[i,0], arr[i,0]+arr[i,9]*scale]
        connection1 = [arr[i,1], arr[i,1]+arr[i,10]*scale]
        connection2 = [arr[i,2], arr[i,2]+arr[i,11]*scale]
        ipv.plot(connection0, connection1, connection2,color="green", lynestyle="--")

        a = arr[i,6:9]
        b = arr[i,9:12]
        n_corss= -np.cross(a,b)
        connection0 = [arr[i,0], arr[i,0]+n_corss[0]]
        connection1 = [arr[i,1], arr[i,1]+n_corss[1]]
        connection2 = [arr[i,2], arr[i,2]+n_corss[2]]
        ipv.plot(connection0, connection1, connection2,color="blue", lynestyle="--")
        ipv.quiver(arr[i:i+1,0], arr[i:i+1,1], arr[i:i+1,2], 
                 -arr[i:i+1,0]+n_corss[0], -arr[i:i+1,1]+n_corss[1], -arr[i:i+1,2]+n_corss[2], 
                   color="blue", size=5)
    ipv.xlim(-1,1);ipv.ylim(-1,1);ipv.zlim(-1,1)
    ipv.show()


def rotate_image(angles, vector):
    # create rotation matrix
    c1 = np.cos(angles[:,0]).reshape(-1,1,1)
    c2 = np.cos(angles[:,1]).reshape(-1,1,1)
    c3 = np.cos(angles[:,2]).reshape(-1,1,1)

    s1 = np.sin(angles[:,0]).reshape(-1,1,1)
    s2 = np.sin(angles[:,1]).reshape(-1,1,1)
    s3 = np.sin(angles[:,2]).reshape(-1,1,1)

    R = np.concatenate([np.concatenate([c1*c2*c3-s1*s3, c1*s3+c2*c3*s1 , -c3*s2],axis=2),\
                    np.concatenate([-c3*s1-c1*c2*s3,    c1*c3-c2*s1*s3 ,   s2*s3],axis=2),\
                    np.concatenate( [c1*s2,             s1*s2          ,   c2],axis=2)],axis=1)

    # rotate previous values
    for i, v in enumerate(vector):
        vector[i] = np.matmul(R,vector[i])

    return vector

def _getXYZ(p1, angles, projection, img_size_scale):
    projection = projection/np.max(projection)
    img = np.zeros((projection.shape[0], projection.shape[1], 3))
    img[:,:,0] = projection
    img[:,:,1] = projection
    img[:,:,2] = projection

    x = np.linspace(-img_size_scale/2, img_size_scale/2, img.shape[0])
    y = np.linspace(-img_size_scale/2, img_size_scale/2, img.shape[1])
    X, Y = np.meshgrid(x, y)
    x_shape = X.shape
    y_shape = Y.shape
    X = X.flatten() 
    Y = Y.flatten()
    
    Z = np.zeros(X.shape)
    vector = np.column_stack((X, Y, Z))
    rotated_vector = rotate_image(angles, vector)
    
    X = rotated_vector[:,0].reshape(x_shape)+p1[0]
    Y = rotated_vector[:,1].reshape(y_shape)+p1[1]
    Z = rotated_vector[:,2].reshape(x_shape)+p1[2]
    return X, Y, Z, img
    
def plot_images(angles, projections, indices=range(3), img_size_scale=0.05):
    arr = RotationMatrix(angles)

    ipv.clear()
    ipv.figure(width=500, height=500)
    indices=indices if indices else range(len(arr))

    for i in indices:
#         ipv.scatter(arr[i:i+1,0], arr[i:i+1,1], arr[i:i+1,2], marker="sphere", color="blue", size=1)
#         ipv.scatter(arr[i,0]+arr[i:i+1,6]*scale, arr[i,1]+arr[i:i+1,7]*scale, arr[i,2]+arr[i:i+1,8]*scale, marker="sphere", color="red", size=1)
#         ipv.scatter(arr[i,0]+arr[i:i+1,9]*scale, arr[i,1]+arr[i:i+1,10]*scale, arr[i,2]+arr[i:i+1,11]*scale, marker="sphere", color="green", size=1)


#         connection0 = [arr[i,0], arr[i,0]+arr[i,6]*scale]
#         connection1 = [arr[i,1], arr[i,1]+arr[i,7]*scale]
#         connection2 = [arr[i,2], arr[i,2]+arr[i,8]*scale]
#         ipv.plot(connection0, connection1, connection2,color="red", lynestyle="--")
#         connection0 = [arr[i,0], arr[i,0]+arr[i,9]*scale]
#         connection1 = [arr[i,1], arr[i,1]+arr[i,10]*scale]
#         connection2 = [arr[i,2], arr[i,2]+arr[i,11]*scale]
#         ipv.plot(connection0, connection1, connection2,color="green", lynestyle="--")

        n_corss= -np.cross(arr[i,6:9],arr[i,9:12])
        connection0 = [arr[i,0], arr[i,0]+n_corss[0]]
        connection1 = [arr[i,1], arr[i,1]+n_corss[1]]
        connection2 = [arr[i,2], arr[i,2]+n_corss[2]]
        ipv.plot(connection0, connection1, connection2,color="blue", lynestyle="--")
        ipv.xlim(-1,1);ipv.ylim(-1,1);ipv.zlim(-1,1)


        X, Y, Z, img = _getXYZ(arr[i], np.array([angles[i]]), projections[i], img_size_scale=img_size_scale)

        ipv.plot_surface(X, Y, Z, color=img)

    
    ipv.show()

def modify_magnitude(angles):
    arr = RotationMatrix(angles)
    
    # two vectors between which the angles will be calculated
    ang_end = arr[:,6:9]
    ang_start = np.array([np.array([1,0,0])*ang_end.shape[0]])
    
    # make unit vector
    ang_start = ang_start/np.linalg.norm(ang_start, axis=1)
    ang_end = ang_end/np.linalg.norm(ang_end, axis=1).reshape(-1, 1)
    
    # magnitude representing the invisible angle rotation
    magnitude = np.arccos(np.clip(np.dot(ang_start, ang_end.T), -1, 1))

    return np.multiply(arr[:,0:3], magnitude.T)

def plot_angles_with_3rd_angle_magnitude(angles):
    ipv.clear()

    new_angles = modify_magnitude(angles)

    ipv.figure(width=500, height=500)
    ipv.scatter(new_angles[:,0], new_angles[:,1], new_angles[:,2], marker="sphere", color="blue", size=1)
    ipv.xlim(-np.pi,np.pi);ipv.ylim(-np.pi,np.pi);ipv.zlim(-np.pi,np.pi)
    ipv.show()


def plot_selected_angles_with_3rd_angle_magnitude(angles, angles_true, indices):
    ipv.clear()

    angles = np.take(angles, indices, axis=0)
    angles_true = np.take(angles_true, indices, axis=0)
    new_angles = modify_magnitude(angles)
    new_angles_true = modify_magnitude(angles_true)

    ipv.figure(width=500, height=500)
    ipv.scatter(new_angles[:,0], new_angles[:,1], new_angles[:,2], marker="sphere", color="red", size=1)
    ipv.scatter(new_angles_true[:,0], new_angles_true[:,1], new_angles_true[:,2], marker="sphere", color="green", size=1)
    
    for i in indices:
        connection0 = [new_angles[i,0], new_angles_true[i,0]]
        connection1 = [new_angles[i,1], new_angles_true[i,1]]
        connection2 = [new_angles[i,2], new_angles_true[i,2]]
        ipv.plot(connection0, connection1, connection2,color="blue", lynestyle="--")
    
    
    ipv.xlim(-np.pi,np.pi);ipv.ylim(-np.pi,np.pi);ipv.zlim(-np.pi,np.pi)
    ipv.show()




##################### Data Info Plots #####################

def plot_angles_count(angles):
    # NOTE: used
    sns.set(style="white", color_codes=True)
    sns.set(style="whitegrid")

    if isinstance(angles[0], tf.Variable):
        angles = np.array([a.numpy() for a in angles])

    for i, ap in enumerate(angles):
        angles[i] = np.array([ap[0]%(2*np.pi), ap[1]%np.pi, ap[2]%(2*np.pi)])
    
    fig, axs = plt.subplots(1, 3, figsize=(17,7))
    axs[0].set_xlim(0,2*np.pi)
    axs[1].set_xlim(0,np.pi)
    axs[2].set_xlim(0,2*np.pi)
    plt.suptitle("Angles Count")

    sns.distplot(angles[:,0], kde=False, bins=40, ax=axs[0], axlabel="Z1 axis angle rotation [rad]", color="r")
    sns.distplot(angles[:,1], kde=False, bins=40, ax=axs[1], axlabel="Y2 axis angle rotation [rad]", color="g")
    sns.distplot(angles[:,2], kde=False, bins=40, ax=axs[2], axlabel="Z3 axis angle rotation [rad]", color="b")
    plt.show()


def plot_distances_count(angles_predicted, angles_true):
    sns.set(style="white", color_codes=True)
    sns.set(style="whitegrid")

    if isinstance(angles_predicted, tf.Tensor):
        angles_predicted = angles_predicted.numpy()
    
    # for i, a in enumerate(angles_predicted):
    #     angles_predicted[i] = np.array([a[0]%(2*np.pi), a[1]%(2*np.pi), a[2]%(2*np.pi)])
        
    # for i, a in enumerate(angles_true):
    #     angles_true[i] = np.array([a[0]%(2*np.pi), a[1]%(2*np.pi), a[2]%(2*np.pi)])
    
    q_true = euler2quaternion(angles_true)
    q_predicted = euler2quaternion(angles_predicted)
    distances = d_q(q_true, q_predicted)
    
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_xlim(0, np.pi)
    plt.suptitle(f"Distances between true and predicted angles\nMEAN={np.mean(distances):.2f} STD={np.std(distances):.2f}")
    s = sns.distplot(distances, kde=False, bins=100, ax=ax, axlabel="Distance [rad]", color="r")
    max_count = int(max([h.get_height() for h in s.patches]))
    #ax.errorbar([np.mean(distances)]*max_count, np.arange(0, max_count, 1), xerr=np.std(distances), fmt='-o', alpha=0.5)
    ax.plot([np.mean(distances)]*max_count, np.arange(0, max_count,1), c="r", lw=4)
    #plt.show()
    return plt

# def plot_distances_count(angles_predicted, angles_true):
#     sns.set(style="white", color_codes=True)
#     sns.set(style="whitegrid")

#     if isinstance(angles_predicted, tf.Tensor):
#         angles_predicted = angles_predicted.numpy()
    
#     q_true = euler2quaternion(angles_true)
#     q_predicted = euler2quaternion(angles_predicted)
#     distances = d_q(q_true, q_predicted)
    
#     fig, ax = plt.subplots(figsize=(10,7))
#     ax.set_xlim(0, np.pi)
# #     ax.set_ylim(0, len(angles_true))
#     plt.suptitle(f"Distances between true and predicted angles\nCNT={len(angles_true)} MEAN={np.mean(distances):.2f} STD={np.std(distances):.2f}")
#     s = sns.distplot(distances, kde=False, bins=100, ax=ax, axlabel="Distance [rad]", color="r")
#     max_count = int(max([h.get_height() for h in s.patches]))
#     #ax.errorbar([np.mean(distances)]*max_count, np.arange(0, max_count, 1), xerr=np.std(distances), fmt='-o', alpha=0.5)
#     ax.plot([np.mean(distances)]*max_count, np.arange(0, max_count,1), c="r", lw=4)
#     #plt.show()
#     return plt


def plot_dP_dQ(dP_values, dQ_values, file_name=None):
    plt.clf()
    # Creating the dataframe for SNS plot
    data = {"d_Q" : dQ_values, #tr_y.numpy(),
            "d_P" : dP_values } #y_tr_pred.T[0]}
    df1 = pd.DataFrame(data=data)

    # Creating the dataframe for SNS plot
    # data = {"d_Q" : val_y.numpy(),
    #         "d_P" : y_val_pred.T[0]}
    # df2 = pd.DataFrame(data=data)

    plt.clf();
    fig, ax = plt.subplots(1, 1, figsize=(6,6));
    sns.scatterplot(x="d_Q", y="d_P", data=df1, color="b", alpha=0.3, label="projection pair", ax=ax);  # "reg", "kde"
    #sns.jointplot(x="d_Q", y="d_P", data=df1, color="b", alpha=0.3, label="projection pair", kind="kde", ax=ax[1]);  # "reg", "kde"
    x = np.arange(0, np.pi);
    sns.regplot(x=x, y=x, color="k", ax=ax)
    if file_name:
        plt.savefig(f"1{file_name}", dpi=150)
    plt.show();

    plt.clf()
    # Creating the dataframe for SNS plot
    data = {"d_Q" : dQ_values, #tr_y.numpy(),
            "d_P" : dP_values } #y_tr_pred.T[0]}
    df1 = pd.DataFrame(data=data)

    # Creating the dataframe for SNS plot
    # data = {"d_Q" : val_y.numpy(),
    #         "d_P" : y_val_pred.T[0]}
    # df2 = pd.DataFrame(data=data)

    plt.clf();
    #sns.scatterplot(x="d_Q", y="d_P", data=df1, color="b", alpha=0.3, label="projection pair", ax=ax[0]);  # "reg", "kde"
    sns.jointplot(x="d_Q", y="d_P", data=df1, color="b", alpha=0.3, label="projection pair", kind="kde");  # "reg", "kde"
    
    if file_name:
        plt.savefig(f"2{file_name}", dpi=150)
    plt.show();

    # variance
    variance = np.sqrt(1/(len(dQ_values)-1)*np.sum(np.power(dP_values-dQ_values, 2)))
    print(f"Variance = {variance}")

    ar_loss = lambda dQ_values, dP_values: tf.reduce_mean(tf.pow((dQ_values - dP_values), 2))
    loss = ar_loss(dQ_values, dP_values).numpy()
    print(f"Min. angle recovery loss possible = {loss}")

    
