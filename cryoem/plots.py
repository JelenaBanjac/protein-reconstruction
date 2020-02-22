import matplotlib.pyplot as plt
import ipyvolume as ipv
from cryoem.projections import RotationMatrix
from scipy.spatial.transform import Rotation as R
import numpy as np
import tensorflow as tf
import seaborn as sns; sns.set(style="white", color_codes=True)
import pandas as pd

fg_color = 'white'
bg_color = 'black'

def _plot(image, title, ax, colorbar=False, mean=0, var=0):
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
        cb.set_label('Closeness', color=fg_color)

        # set colorbar tick color
        cb.ax.yaxis.set_tick_params(color=fg_color)

        # set colorbar edgecolor 
        cb.outline.set_edgecolor(fg_color)

        # set colorbar ticklabels
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)
        
def plot_projection(image, title, mean=0, var=0):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    
    _plot(image, title, ax, colorbar=True)

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

def plot_euclidean_space(angles):
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
    aps = RotationMatrix(np.take(angles, selected, axis=0))
    ats = RotationMatrix(np.take(angles_true, selected, axis=0))
    ipv.scatter(ats[:,0], ats[:,1], ats[:,2], marker="sphere", color="green", size=1)
    ipv.scatter(aps[:,0], aps[:,1], aps[:,2], marker="sphere", color="red", size=1)
    for i in selected:
        connection0 = [ats[i,0], aps[i,0]]
        connection1 = [ats[i,1], aps[i,1]]
        connection2 = [ats[i,2], aps[i,2]]
        ipv.plot(connection0, connection1, connection2,color="red", lynestyle="--")
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
 

def plot_SO3_space(angles, rotation_axes="zyz", normalized=False):
    """ zyz - intrinsic rotation ()
        ZYZ - extrinsic rotation (rotates about axes of motionless coordinate system)
    """
    ipv.clear()

    if isinstance(angles[0], tf.Variable):
        angles = np.array([a.numpy() for a in angles])

    rotation_vectors = R.from_euler(seq=rotation_axes, angles=angles).as_rotvec()
    magnitude = np.linalg.norm(rotation_vectors, axis=1)

    if normalized:
        rotation_vectors /= magnitude[:, np.newaxis]
        ipv.xlim(-1, 1);ipv.ylim(-1,1);ipv.zlim(-1, 1)
    else:
        ipv.xlim(-np.pi, np.pi);ipv.ylim(-np.pi,np.pi);ipv.zlim(-np.pi, np.pi)

    ipv.figure(width=500, height=500)
    ipv.scatter(rotation_vectors[:,0], rotation_vectors[:,1], rotation_vectors[:,2], marker="sphere", color="blue", size=1)
    
    ipv.show()


def plot_one_closest_vs_all_in_SO3_space(angles, closest):
    ipv.clear()
    rotation_vectors = R.from_euler(seq="zyz", angles=angles).as_rotvec()
    magnitude = np.linalg.norm(rotation_vectors, axis=1)

    knn = np.take(rotation_vectors, closest, axis=0)
    all = np.delete(rotation_vectors, closest, 0)

    ipv.figure(width=500, height=400)
    ipv.scatter(knn[:,0], knn[:,1], knn[:,2], marker="diamond", color="red", size=1.5)
    ipv.scatter(all[:,0], all[:,2], all[:,1], marker="sphere", color="blue", size=1)
    ipv.show()

def plot_only_closest_in_SO3_space(angles, closest):
    ipv.clear()
    rotation_vectors = R.from_euler(seq="zyz", angles=angles).as_rotvec()
    magnitude = np.linalg.norm(rotation_vectors, axis=1)

    for i in closest:
        main = np.array([rotation_vectors[i[0]]])
        knn = np.take(rotation_vectors, i[1:], axis=0)
        
        ipv.scatter(main[:,0], main[:,1], main[:,2], marker="diamond", color="red", size=1.5)
        ipv.scatter(knn[:,0], knn[:,1], knn[:,2], marker="sphere", color="blue", size=1)
        
        for c in range(5):
            connection0 = [main[:,0][0], knn[c,0]]
            connection1 = [main[:,1][0], knn[c,1]]
            connection2 = [main[:,2][0], knn[c,2]]
            ipv.plot(connection0, connection1, connection2,color="red", lynestyle="--")
        
    ipv.xlim(-np.pi, np.pi);ipv.ylim(-np.pi,np.pi);ipv.zlim(-np.pi, np.pi)
    ipv.show()

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

##################### Data Info Plots #####################

def plot_angles_count(angles):
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


def plot_dP_dQ(dP_values, dQ_values):

    # Creating the dataframe for SNS plot
    data = {"d_Q" : dQ_values, #tr_y.numpy(),
            "d_P" : dP_values } #y_tr_pred.T[0]}
    df1 = pd.DataFrame(data=data)

    # Creating the dataframe for SNS plot
    # data = {"d_Q" : val_y.numpy(),
    #         "d_P" : y_val_pred.T[0]}
    # df2 = pd.DataFrame(data=data)

    plt.clf();
    fig, ax = plt.subplots(1, 1, figsize=(15,10));
    sns.scatterplot(x="d_Q", y="d_P", data=df1, color="b", alpha=0.3, label="projection pair", ax=ax);  # "reg", "kde"
    #sns.jointplot(x="d_Q", y="d_P", data=df1, color="b", alpha=0.3, label="projection pair", kind="kde", ax=ax[1]);  # "reg", "kde"
    x = np.arange(0, np.pi);
    sns.regplot(x=x, y=x, color="k", ax=ax)
    plt.show();

def plot_dP_dQ_density(dP_values, dQ_values):

    # Creating the dataframe for SNS plot
    data = {"d_Q" : dQ_values, #tr_y.numpy(),
            "d_P" : dP_values } #y_tr_pred.T[0]}
    df1 = pd.DataFrame(data=data)

    # Creating the dataframe for SNS plot
    # data = {"d_Q" : val_y.numpy(),
    #         "d_P" : y_val_pred.T[0]}
    # df2 = pd.DataFrame(data=data)

    plt.clf();
    #fig, ax = plt.subplots(1, 1, figsize=(15,10));
    #sns.scatterplot(x="d_Q", y="d_P", data=df1, color="b", alpha=0.3, label="projection pair", ax=ax[0]);  # "reg", "kde"
    sns.jointplot(x="d_Q", y="d_P", data=df1, color="b", alpha=0.3, label="projection pair", kind="kde");  # "reg", "kde"
    plt.show();