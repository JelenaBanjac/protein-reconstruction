from pyquaternion import Quaternion
import numpy as np

def Q(angle):
    """
    Quaternion implements 3 rotations along x, y, z axis. 
    We compose them to get the final (single) rotation.
    """
    qx = Quaternion(axis=[1, 0, 0], angle=angle[0])
    qy = Quaternion(axis=[0, 1, 0], angle=angle[1])
    qz = Quaternion(axis=[0, 0, 1], angle=angle[2])
    
    # compose rotations above
    q = qx*qy*qz

    return q

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