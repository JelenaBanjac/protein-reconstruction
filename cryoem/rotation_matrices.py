import tensorflow as tf
import numpy as np


def RotationMatrix(angles):
    """ Rotation matrix 

    Rotation matrix from: https://www.geometrictools.com/Documentation/EulerAngles.pdf
    Chapter 2.12. Factor as Rz0 Ry Rz1
    Also, playing with: https://eater.net/quaternions/video/doublecover

    `vectors` has size N x 12, where N is the number of projections and 12 corresponds to the following values:
    ( rayX, rayY, rayZ, dX, dY, dZ, uX, uY, uZ, vX, vY, vZ )
    Where:
    - ray : the ray direction
    - d : the center of the detector
    - u : the vector from detector pixel (0,0) to (0,1)
    - v : the vector from detector pixel (0,0) to (1,0)
    Source: https://www.astra-toolbox.com/docs/geom3d.html
    """
    vectors = np.zeros((angles.shape[0],12))
    vectors[:,0:3] = [0, 0, 1]

    # center of detector
    vectors[:,3:6] = 0
     
    # vector from detector pixel (0,0) to (0,1)
    vectors[:,6:9] = [1, 0, 0]
     
    # vector from detector pixel (0,0) to (1,0)
    vectors[:,9:12]  = [0, 1, 0]
     
    # create rotation matrix
    c1 = np.cos(angles[:,0]).reshape(-1,1,1)
    c2 = np.cos(angles[:,1]).reshape(-1,1,1)
    c3 = np.cos(angles[:,2]).reshape(-1,1,1)
                    
    s1 = np.sin(angles[:,0]).reshape(-1,1,1)
    s2 = np.sin(angles[:,1]).reshape(-1,1,1)
    s3 = np.sin(angles[:,2]).reshape(-1,1,1)
    vector = vectors[0,:]
     
    # Euler angles
    # R = np.concatenate([np.concatenate([c3*c2*c1-s3*s1, c3*c2*s1 + s3*c1, -c3*s2],axis=2),\
    # 				np.concatenate([-s3*c2*c1-c3*s1,-s3*c2*s1+c3*c1 , s3*s2],axis=2),\
    # 				np.concatenate( [s2*c1,          s2*s1          , c2],axis=2)],axis=1)
    R = np.concatenate([np.concatenate([c1*c2*c3-s1*s3, c1*s3+c2*c3*s1 , -c3*s2],axis=2),\
                    np.concatenate([-c3*s1-c1*c2*s3,    c1*c3-c2*s1*s3 ,   s2*s3],axis=2),\
                    np.concatenate( [c1*s2,             s1*s2          ,   c2],axis=2)],axis=1)
    # BT angles
    # R = np.concatenate([np.concatenate([c1*c2, c2*s1, -s2],axis=2),\
    # 				np.concatenate([c1*s2*s3-c3*s1, c1*c3+s1*s2*s3, c2*s3],axis=2),\
    # 				np.concatenate( [s1*s3+c1*c3*s2, c3*s1*s2-c1*s3, c2*c3],axis=2)],axis=1)

    # rotate previous values
    vectors[:,0:3] = np.matmul(R,vector[0:3])
    vectors[:,6:9] = np.matmul(R,vector[6:9])
    vectors[:,9:12] = np.matmul(R,vector[9:12])
    
    return vectors


def euler2matrix(angles):
    """Convert euler angles to rotation matrix"""
    angles = tf.convert_to_tensor(angles)
    c1 = tf.reshape(tf.math.cos(angles[:,0]), (-1,1,1))
    c2 = tf.reshape(tf.math.cos(angles[:,1]), (-1,1,1))
    c3 = tf.reshape(tf.math.cos(angles[:,2]), (-1,1,1))
                
    s1 = tf.reshape(tf.math.sin(angles[:,0]), (-1,1,1))
    s2 = tf.reshape(tf.math.sin(angles[:,1]), (-1,1,1))
    s3 = tf.reshape(tf.math.sin(angles[:,2]), (-1,1,1))
     
    R = tf.concat([ tf.concat([ c1*c2*c3-s1*s3, c1*s3+c2*c3*s1, -c3*s2],axis=2),\
                    tf.concat([-c3*s1-c1*c2*s3, c1*c3-c2*s1*s3,  s2*s3],axis=2),\
                    tf.concat([          c1*s2,          s1*s2,     c2],axis=2)],axis=1)
    return R



def d_r(M1, M2):
    """Distance between 2 rotation matirces"""
    R = M1 @ tf.transpose(M2, perm=[0, 2, 1])
    mid = (tf.linalg.trace(R)-1)/2
    mid = tf.clip_by_value(mid, -1, 1)
    return tf.math.acos(mid)