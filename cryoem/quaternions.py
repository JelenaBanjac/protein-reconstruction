from pyquaternion import Quaternion
import numpy as np

def euler2quaternion(angle):
    """
    Quaternion implements 3 rotations along x, y, z axis. 
    We compose them to get the final (single) rotation.
    """
    # qz1 = Quaternion(axis=[0, 0, 1], angle=angle[0])
    # qy2 = Quaternion(axis=[0, 1, 0], angle=angle[1])
    # qz3 = Quaternion(axis=[0, 0, 1], angle=angle[2])
    # compose rotations above
    # q = qz1*qy2*qz3
    gamma = angle[0]
    beta = angle[1]
    alpha = angle[2]

    e0 = np.cos((gamma+alpha)/2)*np.cos(beta/2)
    e1 = np.cos((gamma-alpha)/2)*np.sin(beta/2)
    e2 = np.sin((gamma-alpha)/2)*np.sin(beta/2)
    e3 = np.sin((gamma+alpha)/2)*np.cos(beta/2)
    q = Quaternion([e0, e1, e2, e3]).normalized
    
    return q

def quaternion2euler(q):
    e0 = q[0]
    e1 = q[1] 
    e2 = q[2]
    e3 = q[3]

    alpha = np.arctan2(e1*e3+e0*e2, -(e2*e3-e0*e1))
    beta = np.arccos(1-2*(e1**2 + e2**2))
    if beta < 0 or beta > np.pi:
        raise ValueError 
    gamma = np.arctan2(e1*e3-e0*e2, e2*e3+e0*e1)

    angles = [gamma, beta, alpha]

    return angles

def d_Q(q1, q2):
    # TODO: still we need to decide the best measure of distance
    # http://kieranwynn.github.io/pyquaternion/#distance-computation
    return Quaternion.distance(q1, q2)

def quaternion2point(q):
    """ Convert Quaternion to point
    
    We convert Qaternion to the point described with x, y, z values in the Cartesian coordinate system.
    From the Qaternion we get axis and angle. The axis is described as unit vector (ux, uy, uz) and the angle is magnitude of vector.
    Using this two information, we can get the x, y, z coordinates of the point described with axis and angle.
    """
    point = np.array(list(map(lambda x: x * q.angle, q.axis)))
    return point
