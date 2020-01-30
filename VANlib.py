import numpy as np
import scipy.linalg as la
import math as m
import cv2

def createRot (RPY, degrees=False):
    '''
    Creates a rotation matrix from roll, pitch, and yaw (passed in as a 3-element 
    thing).  Assumes radians unless degrees=True.  The axes are assumed to be
    as in a camera (east-down-north in the real-world)
    '''
    #print("This is a week 1 function")
    if degrees:
        #i_ = internal
        i_yaw = m.radians(RPY[2])
        i_pitch = m.radians(RPY[1])
        i_roll = m.radians(RPY[0])
    else:
        i_yaw = RPY[2]
        i_pitch = RPY[1]
        i_roll = RPY[0]
        
    R_yaw = np.array([[m.cos(i_yaw), 0., -m.sin(i_yaw)],[0., 1., 0.],[m.sin(i_yaw), 0., m.cos(i_yaw)]])
    R_pitch = np.array([[1., 0., 0.],[0., m.cos(i_pitch), m.sin(i_pitch)],[0., -m.sin(i_pitch), m.cos(i_pitch)]])
    R_roll = np.array([[m.cos(i_roll), m.sin(i_roll), 0.],[-m.sin(i_roll), m.cos(i_roll), 0.],[0., 0., 1.]])
    return np.dot(R_roll, np.dot(R_pitch, R_yaw))

def lineNormalToPoints(n, im_size):
    '''This returns either a tuple or list with two points inside of it.  These
    points should be on the edge of the image and correspond with the normal (n)
    passed in as the first parameter.  i.e. n^T x = 0 where x are the points 
    returned by this function.  The returned points should be 1-d numpy arrays with 
    2 elements'''

    print("This is a week 3 function")

def createP (K, c_R_w, w_t_cam):
    '''Returns a 3x4 P matrix, created from a calibration matrix K,
    the world to camera rotation matrix c_R_w, and the location of the camera in the 
    world coordinate system
    '''
    print("This is a week 3 function")

def projectPoints(P,X):
    '''X is a 3xN or 4xN array with N points in it.  P is the projection matrix.
    Note that this function assumes the 4th element of X (if it is 4xN) are all
    1's.  If not, I don't check and who knows what happens... '''
    print("This is a week 3 function")

def fundamentalMatrixFromGeometry(K, c1_R_w, w_t_cam1, c2_R_w, w_t_cam2):
    '''
    This function assumes two cameras, both with the same calibration matrix.  
    It should return the fundamental matrix that takes points in camera 1 image space
    and returns a line in camera 2's image space.  The rotation and location
    of the two cameras are with respect to a "world" frame as denoted by the
    parameter notations
    '''
    print("This is a week 3 function")

def backprojectPoints(K, c_R_w, w_t_cam, x_dist_tuples):
    '''  
    This function takes in a list of tuples containing an x (image) point and
    a "z" distance.  It then uses K, R, and t to project a point in space.  It
    will return a numpy array of world (3-D) points size 3xN, where N is the 
    length of the input list
    '''
    print("This is a week 3 function")
        
def rotateCameraAtPoint( w_point, w_camera, random_roll=False ):
    '''
    This function computes a yaw and pitch to point the center of a camera at
    w_camera at the point at w_point.  w_ means both are in the world coordinate.
    It will return a rotation matrix c_R_w.  If random_roll is true, then a random
    roll will be added to the DCM

    Math-wise, the center image vector ([0,0,1]) gets rotated with yaw (y) and 
    pitch(y) to become:
    [sin(y)cos(p), -sin(p), cos(y)cos(p)]
    HINT: Use atan2 as opposed to atan
    '''
    print("This is a week 4 function")

def backupCamera( w_points, c_R_w, w_camera, K, im_size, pixel_buffer=None ):
    '''
    This function takes in an array of points in the world coordinate system 
    (w_points, 3xN) and "backs up" the camera location, where the camera is oriented 
    according to c_R_w, until all of the points are in the FoV of the camera.
    The FoV is determined by the K and im_size (width, height) passed in.

    If pixel_buffer is passed in, it will back up the camera further so that no
    pixel is within pixel_buffer pixels of the edge of the image
    '''
    print("This is a week 4 function")

def imPointDerivX(P, X, row=None):
    '''
    This takes the derivative of u or v (row = 0 or 1) w.r.t. X in the formula P * X.
    It is a bit more tricky because it is a divide.  Using mixed LaTeX and Python notation,
    u = \frac{P[0,:]X}{P[2,:]X} and v = \frac{P[1,:]X}{P[2,:]X}.  This function
    basically takes the derivative of that divide assuming P is fixed.

    Note that I assume X is a 3 element unit, implying a 4th element =1 when computing the
    derivative.  While projective geometry says a 0 in the 4th element should be possible 
    (making a point at infinity), I ignore that possibility.  This also means the return matrix
    has 3 columns.

    If row is None (the default), than a 2x3 matrix is returned as both rows will be returned.
    '''
    print("This is a Week 4 function")

def refineTriangulate(Px_list):
    '''  
    This function will take a list of projection matrices and their
    corresponding pixel locations and find the best estimate of the point that
    is represented by all those image locations.  We assume that noise _in the image plane_
    is the same across all images and isotropic within the image.  This uses a Gauss-Newton
    optimization procedure to to find the point in 3-D space. To initialize, it will use 
    the first and last P,x tuple in openCV's triangulate points.

    'Px_list' is an array or list of tuples.  Each tuple consists of 
    a 3x4 P matrix (createP is useful here) and a pixel location in that image.  This
    list must have at least 2 entries

    '''
    print("This is a Week 4 function")

    num_pts = len(Px_list)
    assert num_pts > 1, "Need at least 2 projection matrices and x values to triangulate"

    