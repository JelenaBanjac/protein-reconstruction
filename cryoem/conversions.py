import numpy as np
import tensorflow as tf
from tensorflow_graphics.util import safe_ops, asserts, shape
from tensorflow_graphics.math import vector
from tensorflow_graphics.geometry.transformation import quaternion, euler
import math

def distance_difference(angles_predicted, angles_true):
    q_predicted = euler2quaternion(angles_predicted)
    q_true = euler2quaternion(angles_true)
    qd = np.mean(d_q(q_predicted, q_true).numpy())
    print(f"Mean `quaternion` distance between true and predicted values: {qd:.3f} rad ({np.degrees(qd):.3f} degrees)")

    # R_predicted = euler2matrix(angles_predicted)
    # R_true = euler2matrix(angles_true)
    # rd = np.mean(d_r(R_predicted, R_true).numpy())
    # print(f"Mean `rotation matrix` distance between true and predicted values: {rd:.3f} rad ({np.degrees(rd):.3f} degrees)")

    return qd #, rd

def euler2quaternion(angles, transposed=True):
    
    angles = tf.convert_to_tensor(value=angles)

    shape.check_static(tensor=angles, tensor_name="angles", has_dim_equals=(-1, 3))
    
    theta_z1, theta_y, theta_z0 = tf.unstack(angles, axis=-1)

    # create rotation matrix
    c1 = tf.cos(theta_z1)
    c2 = tf.cos(theta_y)
    c3 = tf.cos(theta_z0)

    s1 = tf.sin(theta_z1)
    s2 = tf.sin(theta_y)
    s3 = tf.sin(theta_z0)

    if not transposed:
        r00 = c1*c2*c3-s1*s3
        r10 = c1*s3+c2*c3*s1
        r20 = -c3*s2
        r01 = -c3*s1-c1*c2*s3
        r11 = c1*c3-c2*s1*s3
        r21 = s2*s3
        r02 = c1*s2
        r12 = s1*s2 
        r22 = c2

    else:
        # PROJECTIONS CODE
        r00 = c1*c2*c3-s1*s3
        r01 = c1*s3+c2*c3*s1
        r02 = -c3*s2
        r10 = -c3*s1-c1*c2*s3
        r11 = c1*c3-c2*s1*s3
        r12 = s2*s3
        r20 = c1*s2
        r21 = s1*s2 
        r22 = c2

    w2 = 1/4*(1+ r00 + r11 + r22)
    w2_is_pos = tf.greater(w2, 0)
    
    x2 = -1/2*(r11+r22)
    x2_is_pos = tf.greater(x2, 0)
    
    y2 = 1/2*(1-r22)
    y2_is_pos = tf.greater(y2, 0)
    
    w = tf.compat.v1.where(w2_is_pos, tf.sqrt(w2), tf.zeros_like(w2))
    x = tf.compat.v1.where(w2_is_pos, 1/(4*w)*(r21-r12),
                                        tf.compat.v1.where(x2_is_pos, tf.sqrt(x2), tf.zeros_like(x2)))
    y = tf.compat.v1.where(w2_is_pos, 1/(4*w)*(r02-r20),
                                        tf.compat.v1.where(x2_is_pos, r01/(2*x), 
                                                                    tf.compat.v1.where(y2_is_pos, tf.sqrt(y2), tf.zeros_like(y2))))
    
    z = tf.compat.v1.where(w2_is_pos, 1/(4*w)*(r10-r01), 
                                        tf.compat.v1.where(x2_is_pos, r02/(2*x), 
                                                                    tf.compat.v1.where(y2_is_pos, r12/(2*y), tf.ones_like(y2))))
    
    return tf.stack((x, y, z, w), axis=-1)

def quaternion2euler(quaternions, transposed=True):
    """https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/euler.py"""
    
    def general_case(r02, r12, r20, r21, r22, eps_addition):
        """Handles the general case."""
        theta_y = tf.acos(r22)
        #sign_sin_theta_y = safe_ops.nonzero_sign(tf.sin(theta_y))
        
        r02 = safe_ops.nonzero_sign(r02) * eps_addition + r02
        r22 = safe_ops.nonzero_sign(r22) * eps_addition + r22
        
        theta_z0 = tf.atan2(r12, r02)
        theta_z1 = tf.atan2(r21, -r20)
        return tf.stack((theta_z0, theta_y, theta_z1), axis=-1)

    def gimbal_lock(r22, r11, r10, eps_addition):
        """Handles Gimbal locks.
        It is gimbal when r22 is -1 or 1"""
        sign_r22 = safe_ops.nonzero_sign(r22)
        r11 = safe_ops.nonzero_sign(r11) * eps_addition + r11
        
        theta_z0 = tf.atan2(sign_r22 * r10, r11)
        
        theta_y = tf.constant(math.pi/2.0, dtype=r20.dtype) - sign_r22 * tf.constant(math.pi/2.0, dtype=r20.dtype)
        theta_z1 = tf.zeros_like(theta_z0)
        angles = tf.stack((theta_z0, theta_y, theta_z1), axis=-1)
        return angles

    with tf.compat.v1.name_scope(None, "euler_from_quaternion", [quaternions]):
        quaternions = tf.convert_to_tensor(value=quaternions)

        shape.check_static(
            tensor=quaternions,
            tensor_name="quaternions",
            has_dim_equals=(-1, 4))

        x, y, z, w = tf.unstack(quaternions, axis=-1)
        tx = safe_ops.safe_shrink(2.0 * x, -2.0, 2.0, True)
        ty = safe_ops.safe_shrink(2.0 * y, -2.0, 2.0, True)
        tz = safe_ops.safe_shrink(2.0 * z, -2.0, 2.0, True)
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z

        # The following is clipped due to numerical instabilities that can take some
        # enties outside the [-1;1] range.
        
        if not transposed:
            r00 = safe_ops.safe_shrink(1.0 - (tyy + tzz), -1.0, 1.0, True)
            r01 = safe_ops.safe_shrink(txy - twz, -1.0, 1.0, True)
            r02 = safe_ops.safe_shrink(txz + twy, -1.0, 1.0, True)

            r10 = safe_ops.safe_shrink(txy + twz, -1.0, 1.0, True)
            r11 = safe_ops.safe_shrink(1.0 - (txx + tzz), -1.0, 1.0, True)
            r12 = safe_ops.safe_shrink(tyz - twx, -1.0, 1.0, True)

            r20 = safe_ops.safe_shrink(txz - twy, -1.0, 1.0, True)
            r21 = safe_ops.safe_shrink(tyz + twx, -1.0, 1.0, True)
            r22 = safe_ops.safe_shrink(1.0 - (txx + tyy), -1.0, 1.0, True)
        
        else:
            r00 = safe_ops.safe_shrink(1.0 - (tyy + tzz), -1.0, 1.0, True)
            r01 = safe_ops.safe_shrink(txy + twz, -1.0, 1.0, True)
            r02 = safe_ops.safe_shrink(txz - twy, -1.0, 1.0, True)

            r10 = safe_ops.safe_shrink(txy - twz, -1.0, 1.0, True)
            r11 = safe_ops.safe_shrink(1.0 - (txx + tzz), -1.0, 1.0, True)
            r12 = safe_ops.safe_shrink(tyz + twx, -1.0, 1.0, True)

            r20 = safe_ops.safe_shrink(txz + twy, -1.0, 1.0, True)
            r21 = safe_ops.safe_shrink(tyz - twx, -1.0, 1.0, True)
            r22 = safe_ops.safe_shrink(1.0 - (txx + tyy), -1.0, 1.0, True)
        
        eps_addition = asserts.select_eps_for_addition(quaternions.dtype)
        general_solution = general_case(r02, r12, r20, r21, r22, eps_addition)
        gimbal_solution = gimbal_lock(r22, r11, r10, eps_addition)
        
        # The general solution is unstable close to the Gimbal lock, and the gimbal
        # solution is not toooff in these cases.
        # Check if r22 is 1 or -1
        is_gimbal = tf.less(tf.abs(tf.abs(r22) - 1.0), 1.0e-6)
        gimbal_mask = tf.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)
        
        return tf.compat.v1.where(gimbal_mask, gimbal_solution, general_solution)          
    
def d_q(q1, q2):
    q1 = tf.cast(tf.convert_to_tensor(value=q1), dtype=tf.float64)
    q2 = tf.cast(tf.convert_to_tensor(value=q2), dtype=tf.float64)
    
    shape.check_static(tensor=q1, tensor_name="quaternion1", has_dim_equals=(-1, 4))
    shape.check_static(tensor=q2, tensor_name="quaternion2", has_dim_equals=(-1, 4))

    q1 = quaternion.normalize(q1)
    q2 = quaternion.normalize(q2)
    
    dot_product = vector.dot(q1, q2, keepdims=False)
    
    # Ensure dot product is in range [-1. 1].
    eps_dot_prod = 1.8 * asserts.select_eps_for_addition(dot_product.dtype)
    dot_product = safe_ops.safe_shrink(dot_product, -1, 1, open_bounds=False, eps=eps_dot_prod)

    return 2.0 * tf.acos(tf.abs(dot_product)) 

def quaternion2matrix(quaternions, transposed=True):
    """https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/euler.py"""

    quaternions = tf.convert_to_tensor(value=quaternions)

    shape.check_static(
        tensor=quaternions,
        tensor_name="quaternions",
        has_dim_equals=(-1, 4))

    x, y, z, w = tf.unstack(quaternions, axis=-1)
    tx = safe_ops.safe_shrink(2.0 * x, -2.0, 2.0, True)
    ty = safe_ops.safe_shrink(2.0 * y, -2.0, 2.0, True)
    tz = safe_ops.safe_shrink(2.0 * z, -2.0, 2.0, True)
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    # The following is clipped due to numerical instabilities that can take some
    # enties outside the [-1;1] range.
    
    if not transposed:
        r00 = safe_ops.safe_shrink(1.0 - (tyy + tzz), -1.0, 1.0, True)
        r01 = safe_ops.safe_shrink(txy - twz, -1.0, 1.0, True)
        r02 = safe_ops.safe_shrink(txz + twy, -1.0, 1.0, True)

        r10 = safe_ops.safe_shrink(txy + twz, -1.0, 1.0, True)
        r11 = safe_ops.safe_shrink(1.0 - (txx + tzz), -1.0, 1.0, True)
        r12 = safe_ops.safe_shrink(tyz - twx, -1.0, 1.0, True)

        r20 = safe_ops.safe_shrink(txz - twy, -1.0, 1.0, True)
        r21 = safe_ops.safe_shrink(tyz + twx, -1.0, 1.0, True)
        r22 = safe_ops.safe_shrink(1.0 - (txx + tyy), -1.0, 1.0, True)
    
    else:
        r00 = safe_ops.safe_shrink(1.0 - (tyy + tzz), -1.0, 1.0, True)
        r01 = safe_ops.safe_shrink(txy + twz, -1.0, 1.0, True)
        r02 = safe_ops.safe_shrink(txz - twy, -1.0, 1.0, True)

        r10 = safe_ops.safe_shrink(txy - twz, -1.0, 1.0, True)
        r11 = safe_ops.safe_shrink(1.0 - (txx + tzz), -1.0, 1.0, True)
        r12 = safe_ops.safe_shrink(tyz + twx, -1.0, 1.0, True)

        r20 = safe_ops.safe_shrink(txz + twy, -1.0, 1.0, True)
        r21 = safe_ops.safe_shrink(tyz - twx, -1.0, 1.0, True)
        r22 = safe_ops.safe_shrink(1.0 - (txx + tyy), -1.0, 1.0, True)
        
    R = tf.stack((r00, r10, r20, r01, r11, r21, r02, r12, r22), axis=-1)
    R = tf.reshape(R, (-1, 3, 3))
    
    return R

def matrix2quaternion(R, transposed=True):
    R = tf.convert_to_tensor(value=R)
    
    R = tf.reshape(R, [-1, 9])
    if not transposed:
        r00, r10, r20, r01, r11, r21, r02, r12, r22 = tf.unstack(R, axis=-1)
        
    else:
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = tf.unstack(R, axis=-1)
    

    w2 = 1/4*(1+ r00 + r11 + r22)
    w2_is_pos = tf.greater(w2, 0)
    
    x2 = -1/2*(r11+r22)
    x2_is_pos = tf.greater(x2, 0)
    
    y2 = 1/2*(1-r22)
    y2_is_pos = tf.greater(y2, 0)
    
    w = tf.compat.v1.where(w2_is_pos, tf.sqrt(w2), tf.zeros_like(w2))
    x = tf.compat.v1.where(w2_is_pos, 1/(4*w)*(r21-r12),
                                        tf.compat.v1.where(x2_is_pos, tf.sqrt(x2), tf.zeros_like(x2)))
    y = tf.compat.v1.where(w2_is_pos, 1/(4*w)*(r02-r20),
                                        tf.compat.v1.where(x2_is_pos, r01/(2*x), 
                                                                    tf.compat.v1.where(y2_is_pos, tf.sqrt(y2), tf.zeros_like(y2))))
    
    z = tf.compat.v1.where(w2_is_pos, 1/(4*w)*(r10-r01), 
                                        tf.compat.v1.where(x2_is_pos, r02/(2*x), 
                                                                    tf.compat.v1.where(y2_is_pos, r12/(2*y), tf.ones_like(y2))))
    
    return tf.stack((x, y, z, w), axis=-1)

def euler2matrix(angles, transposed=True):
    
    angles = tf.convert_to_tensor(value=angles, dtype=tf.float64)

    theta_z1, theta_y, theta_z0 = tf.unstack(angles, axis=-1)

    # create rotation matrix
    c1 = tf.cos(theta_z1)
    c2 = tf.cos(theta_y)
    c3 = tf.cos(theta_z0)

    s1 = tf.sin(theta_z1)
    s2 = tf.sin(theta_y)
    s3 = tf.sin(theta_z0)

    if not transposed:
        r00 = c1*c2*c3-s1*s3
        r10 = c1*s3+c2*c3*s1
        r20 = -c3*s2
        r01 = -c3*s1-c1*c2*s3
        r11 = c1*c3-c2*s1*s3
        r21 = s2*s3
        r02 = c1*s2
        r12 = s1*s2 
        r22 = c2

    else:
        # PROJECTIONS CODE
        r00 = c1*c2*c3-s1*s3
        r01 = c1*s3+c2*c3*s1
        r02 = -c3*s2
        r10 = -c3*s1-c1*c2*s3
        r11 = c1*c3-c2*s1*s3
        r12 = s2*s3
        r20 = c1*s2
        r21 = s1*s2 
        r22 = c2

    R = tf.stack((r00, r10, r20, r01, r11, r21, r02, r12, r22), axis=-1)
    R = tf.reshape(R, (-1, 3, 3))
    
    return R

def matrix2euler(R, transposed=True):
    """https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/euler.py"""
    
    def general_case(r02, r12, r20, r21, r22, eps_addition):
        """Handles the general case."""
        theta_y = tf.acos(r22)
        #sign_sin_theta_y = safe_ops.nonzero_sign(tf.sin(theta_y))
        
        r02 = safe_ops.nonzero_sign(r02) * eps_addition + r02
        r22 = safe_ops.nonzero_sign(r22) * eps_addition + r22
        
        theta_z0 = tf.atan2(r12, r02)
        theta_z1 = tf.atan2(r21, -r20)
        return tf.stack((theta_z0, theta_y, theta_z1), axis=-1)

    def gimbal_lock(r22, r11, r10, eps_addition):
        """Handles Gimbal locks.
        It is gimbal when r22 is -1 or 1"""
        sign_r22 = safe_ops.nonzero_sign(r22)
        r11 = safe_ops.nonzero_sign(r11) * eps_addition + r11
        
        theta_z0 = tf.atan2(sign_r22 * r10, r11)
        
        theta_y = tf.constant(math.pi/2.0, dtype=r20.dtype) - sign_r22 * tf.constant(math.pi/2.0, dtype=r20.dtype)
        theta_z1 = tf.zeros_like(theta_z0)
        angles = tf.stack((theta_z0, theta_y, theta_z1), axis=-1)
        return angles

    R = tf.convert_to_tensor(value=R)
        
    R = tf.reshape(R, [-1, 9])
    if not transposed:
        r00, r10, r20, r01, r11, r21, r02, r12, r22 = tf.unstack(R, axis=-1)
        
    else:
        r00, r01, r02, r10, r11, r12, r20, r21, r22 = tf.unstack(R, axis=-1)
    
    eps_addition = asserts.select_eps_for_addition(R.dtype)
    general_solution = general_case(r02, r12, r20, r21, r22, eps_addition)
    gimbal_solution = gimbal_lock(r22, r11, r10, eps_addition)
    
    # The general solution is unstable close to the Gimbal lock, and the gimbal
    # solution is not toooff in these cases.
    # Check if r22 is 1 or -1
    is_gimbal = tf.less(tf.abs(tf.abs(r22) - 1.0), 1.0e-6)
    gimbal_mask = tf.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)
    
    return tf.compat.v1.where(gimbal_mask, gimbal_solution, general_solution)     
