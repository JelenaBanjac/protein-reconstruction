import numpy as np
import tensorflow as tf
from tensorflow_graphics.util import safe_ops, asserts, shape
from tensorflow_graphics.math import vector
from tensorflow_graphics.geometry.transformation import quaternion, euler
import math

def euler2quaternion(angles):
    """
    Document: https://www.sedris.org/wg8home/Documents/WG80485.pdf
    Also:pg.25 https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770019231.pdf
    
    Tait-Bryan angles:
        Quaternion implements 3 rotations along z-y-x axis. 
        We compose them to get the final (single) rotation.
    Euler angles:
        Quaternion implements 3 rotations along z-y-z axis.
    General:
        q = xi + yj + zk + w
    Stored as 4D:
        [x, y, z, w].T
    """
    with tf.compat.v1.name_scope(None, "quaternion_from_euler", [angles]):
        #print(angles)
        a = [angles[i] for i in range(len(angles))]

        a = tf.convert_to_tensor(value=a)

        shape.check_static(tensor=a, tensor_name="angles", has_dim_equals=(-1, 3))

        half_angles = a / 2.0
        cos_half_angles = tf.cos(half_angles)
        sin_half_angles = tf.sin(half_angles)

        c3, c2, c1 = tf.unstack(cos_half_angles, axis=-1)
        s3, s2, s1 = tf.unstack(sin_half_angles, axis=-1)
        # Tait-Bryan angles
        #w = c1 * c2 * c3 + s1 * s2 * s3
        #x = -c1 * s2 * s3 + s1 * c2 * c3
        #y = c1 * s2 * c3 + s1 * c2 * s3
        #z = -s1 * s2 * c3 + c1 * c2 * s3
        
        # Euler angles
        w = c1*c2*c3 - s1*c2*s3
        x = c1*s2*s3 - s1*s2*c3
        y = c1*s2*c3 + s1*s2*s3
        z = c1*c2*s3 + s1*c2*c3
        return tf.stack((x, y, z, w), axis=-1)

def quaternion2euler(quaternions):
    """https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/euler.py"""
    def general_case(r02, r12, r20, r21, r22, eps_addition):
        """Handles the general case."""
#         theta_y = -tf.asin(r20)
#         sign_cos_theta_y = safe_ops.nonzero_sign(tf.cos(theta_y))
#         r00 = safe_ops.nonzero_sign(r00) * eps_addition + r00
#         r22 = safe_ops.nonzero_sign(r22) * eps_addition + r22
#         theta_z = tf.atan2(r10 * sign_cos_theta_y, r00 * sign_cos_theta_y)
#         theta_x = tf.atan2(r21 * sign_cos_theta_y, r22 * sign_cos_theta_y)
#         return tf.stack((theta_x, theta_y, theta_z), axis=-1)
        theta_y = tf.acos(r22)
        # TODO: check this >>>
        sign_sin_theta_y = safe_ops.nonzero_sign(tf.sin(theta_y))
        
        r02 = safe_ops.nonzero_sign(r02) * eps_addition + r02
        r22 = safe_ops.nonzero_sign(r22) * eps_addition + r22
        theta_z0 = tf.atan2(r12 * sign_sin_theta_y, r02 * sign_sin_theta_y)
        theta_z1 = tf.atan2(r21 * sign_sin_theta_y, -r20 * sign_sin_theta_y)
        return tf.stack((theta_z0, theta_y, theta_z1), axis=-1)

    def gimbal_lock(r22, r11, r10, eps_addition):
        """Handles Gimbal locks."""
#         sign_r20 = safe_ops.nonzero_sign(r20)
#         r02 = safe_ops.nonzero_sign(r02) * eps_addition + r02
#         theta_x = tf.atan2(-sign_r20 * r01, -sign_r20 * r02)
#         theta_y = -sign_r20 * tf.constant(math.pi / 2.0, dtype=r20.dtype)
#         theta_z = tf.zeros_like(theta_x)
#         angles = tf.stack((theta_x, theta_y, theta_z), axis=-1)
#         return angles
        sign_r22 = safe_ops.nonzero_sign(r22)
        r11 = safe_ops.nonzero_sign(r11) * eps_addition + r11
        theta_z0 = tf.atan2(-sign_r22 * r10, -sign_r22 * r11)  # TODO: was -
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
        r00 = safe_ops.safe_shrink(1.0 - (tyy + tzz), -1.0, 1.0, True)
        r10 = safe_ops.safe_shrink(txy + twz, -1.0, 1.0, True)
        r21 = safe_ops.safe_shrink(tyz + twx, -1.0, 1.0, True)
        r22 = safe_ops.safe_shrink(1.0 - (txx + tyy), -1.0, 1.0, True)
        r20 = safe_ops.safe_shrink(txz - twy, -1.0, 1.0, True)
        r01 = safe_ops.safe_shrink(txy - twz, -1.0, 1.0, True)
        r02 = safe_ops.safe_shrink(txz + twy, -1.0, 1.0, True)
        
        r12 = safe_ops.safe_shrink(tyz - twx, -1.0, 1.0, True)
        r11 = safe_ops.safe_shrink(1.0 - (txx + tzz), -1.0, 1.0, True)
        eps_addition = asserts.select_eps_for_addition(quaternions.dtype)
        general_solution = general_case(r02, r12, r20, r21, r22, eps_addition)
        gimbal_solution = gimbal_lock(r22, r11, r10, eps_addition)
        
        # The general solution is unstable close to the Gimbal lock, and the gimbal
        # solution is not toooff in these cases.
        is_gimbal = tf.less(tf.abs(tf.abs(r22) - 1.0), 1.0e-6)
        gimbal_mask = tf.stack((is_gimbal, is_gimbal, is_gimbal), axis=-1)
        
        return tf.compat.v1.where(gimbal_mask, gimbal_solution, general_solution)        
    
def d_q(q1, q2):
     with (tf.compat.v1.name_scope(None, "quaternion_relative_angle",[q1, q2])):
        q1 = tf.convert_to_tensor(value=q1)
        q2 = tf.convert_to_tensor(value=q2)
      
        shape.check_static(
            tensor=q1, tensor_name="quaternion1", has_dim_equals=(-1, 4))
        shape.check_static(
            tensor=q2, tensor_name="quaternion2", has_dim_equals=(-1, 4))

        q1 = quaternion.normalize(q1)
        q2 = quaternion.normalize(q2)
        
        dot_product = vector.dot(q1, q2, keepdims=False)
        
        # Ensure dot product is in range [-1. 1].
        const = 1.8 #4.0 #.63
        eps_dot_prod = const * asserts.select_eps_for_addition(dot_product.dtype)
        dot_product = safe_ops.safe_shrink(
            dot_product, -1, 1, open_bounds=False, eps=eps_dot_prod)

        return 2.0 * tf.acos(tf.abs(dot_product)) 


# def d_p(p1, p2):
#     # (learned) distance between two images.
#     # for now, Euclid dist
#     p1 = tf.convert_to_tensor(value=p1, dtype=np.float64)
#     p2 = tf.convert_to_tensor(value=p2, dtype=np.float64)

#     if len(p1.shape) > 1:
#         dist = tf.norm(p1-p2, ord='euclidean', axis=1, keepdims=True)
#     else:
#         dist = tf.norm(p1-p2, ord='euclidean')

#     return dist