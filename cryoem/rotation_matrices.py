import tensorflow as tf
import numpy as np

def euler2matrix(angles):
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