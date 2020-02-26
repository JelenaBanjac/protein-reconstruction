def rotate_2d(angles, vector):
    # create rotation matrix

    c1 = np.cos(angles[0]).reshape(-1,1)

    s1 = np.sin(angles[0]).reshape(-1,1)

    R = np.concatenate([np.concatenate([c1, s1 ],axis=1),\
                    np.concatenate([-s1,    c1 ],axis=1)],axis=0)

    # rotate previous values
    for i, v in enumerate(vector):
        vector[i] = np.matmul(R,vector[i])

    return vector

def get_angle(_x, _y, _z):
    # two vectors between which the angles will be calculated
    ang_end = np.array([_x, _y, _z]).T #arr[:,6:9]
    ang_start = np.array([1,0,0])
    
    # make unit vector
    print(ang_start)
    print(ang_end)
    ang_start = ang_start/np.linalg.norm(ang_start)
    ang_end = ang_end/np.linalg.norm(ang_end).reshape(-1, 1)
    
    # magnitude representing the invisible angle rotation
    anlge = np.arccos(np.clip(np.dot(ang_start, ang_end.T), -1, 1))
    
    return anlge

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from matplotlib._png import read_png


def plot_images_equirectangular(angles_true, projections, indices=range(10), img_size_scale=0.1):
        
#     def imscatter(x, y, image, ax=None, zoom=0.1):
        
    
    arr = RotationMatrix(angles_true)#[:,:,3]
    x = arr[:,0]
    y = arr[:,1]
    z = arr[:,2]
    _x = arr[:,6]
    _y = arr[:,7]
    _z = arr[:,8]
    
    _, lat, lon = cartesian_to_spherical(x, y, z)
    lat = np.array(lat)
    lon = np.array(lon)
    
    x1, y1, _,_ = utm.from_latlon(lat, lon)

    x1 = np.array(x1)
    y1 = np.array(y1)
    
#     fig, ax = plt.subplots(figsize=(20,10))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in indices:
        #imscatter(x1[i],y1[i],projections[i],ax=ax)
        
        projection = projections[i]/np.max(projections[i])
        img = np.zeros((projection.shape[0], projection.shape[1], 4))
        img[:,:,0] = projection
        img[:,:,1] = projection
        img[:,:,2] = projection
        img[:,:,3] = np.ones(projection.shape)
        x = np.linspace(-img_size_scale/2, img_size_scale/2, img.shape[0])
        y = np.linspace(-img_size_scale/2, img_size_scale/2, img.shape[1])
        X, Y = np.meshgrid(x, y)
        x_shape = X.shape
        y_shape = Y.shape
        X = X.flatten() 
        Y = Y.flatten()

        vector = np.column_stack((X, Y))
        angle = get_angle(_x[i], _y[i], _z[i])
        rotated_vector = rotate_2d(np.array([angle]), vector)
        
        X = rotated_vector[:,0].reshape(x_shape)+x1[i]
        Y = rotated_vector[:,1].reshape(y_shape)+y1[i]
        print(X.shape)
        print(img.shape)
        Z = np.zeros(X.shape)
    
        img = img.reshape(-1, 4)
        print(img.shape)
        ax.plot_surface(X, Y, Z, color=img)
        
        ax.scatter(x1[i], y1[i], marker="o", color="blue")
    plt.show()