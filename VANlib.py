import numpy as np
import scipy.linalg as la
import math as m
import cv2

def assert_np_matrix(m, shape):
    '''Verify that the variable is an np.ndarray, has the correct dimensionality, 
    and is of the correct shape. Raise an exception if it fails any test.

    Args:
        m: Matrix to test
        shape (tuple(int,int)) or (tuple(int,)): Required shape (rows, cols). 
            If one element is 0, ignores that dimension (allows for a 3XN array, where
            you don't know what N is at compile-time)

    Raises:
        ValueError: Wrong type, dimensionality or size
    '''

    if not type(m) is np.ndarray:
        raise ValueError('Wrong type', 'matrix must be np.ndarray')

    if not len(shape) == len(m.shape):
        raise ValueError('Wrong dimensions', 
                         f'array must be {len(shape)}-dimensional, got {len(m.shape)}-d array')

    if len(shape)==1:
        if shape[0]!=0 and m.shape[0] != shape[0]:
            raise ValueError('Wrong shape', f'matrix must be a {shape} but is {m.shape}')
    else:
        if not shape[0] and not m.shape[1] == shape[1]:
            raise ValueError('Wrong shape', f'matrix must be a {shape} but is {m.shape}')
        
        if not shape[1] and not m.shape[0] == shape[0]:
            raise ValueError('Wrong shape', f'matrix must be a {shape} but is {m.shape}')
        
        if shape[0] and shape[1] and not m.shape == shape:
            #print('ERROR: set_translation(): w_T_c must be a 1x3 matrix')
            raise ValueError('Wrong shape', f'matrix must be a {shape} but is {m.shape}')

def assert_np_matrix_choice(m, shapes):
    '''
    Take in m and see if it is one of a list of shapes

    Args:  shapes is a list of tuples describing shapes.  Each
    must be either 1-D or 2-D (rows, cols)
    '''
    if not type(m) is np.ndarray:
        raise ValueError('Wrong type', 'matrix must be np.ndarray')

    passed = False
    for shape in shapes:
        try:
            assert_np_matrix(m, shape)
        except ValueError:
            pass
        else:
            passed = True
    if not passed:
        raise ValueError('Wrong shape', f'matrix is {m.shape}, but needs to be one of {shapes}')

def createSkewSymmMat ( axis_rots ) -> np.ndarray( (3,3) ):
    '''
    Creates a 3x3 skew symmetric matrix from a 3-vector

    Args:  axis_rots can be a list, tuple, or ndarray of one dimension.
    Anything that allows it to reference its elements as axis_rots[0],
    axis_rots[1], and axis_rots[2]
    '''
    assert(len(axis_rots)==3)
    return np.array([[0., axis_rots[2], -axis_rots[1]],
                     [-axis_rots[2], 0., axis_rots[0]], 
                     [axis_rots[1], -axis_rots[0], 0.]])

def createRot (RPY, degrees=False):
    '''
    Creates a rotation matrix from roll, pitch, and yaw (passed in as a 3-element 
    thing).  Assumes radians unless degrees=True.  The axes are assumed to be
    as in a camera (east-down-north in the real-world)

    RPY can be a list, tuple, or ndarray of one dimension.
    Anything that allows it to reference its elements as RPY[0],
    RPY[1], and RPY[2]
    '''
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

def inImage( pt, im_size, buffer=None, my_eps = 1E-10 ):
    ''' A quick binary check to decide if pt (a pixel location) is within the image
    of size im_size (width,height).  Pt can be any data structure that has [0] and [1]
    as the way to get the x and y location.
    '''
    
    assert_np_matrix_choice(pt, [(2,), (3,)] )
    #Make sure, if a 3-vector, the 3rd element=1
    assert len(pt)==2 or pt[2]==1.
    
    assert(len(im_size)==2)
    
    if buffer==None:
        buffer=-my_eps
    else:
        buffer -= my_eps
    going_out = pt[0] >=buffer and (pt[0] <= (im_size[0]-buffer)) and pt[1] >= buffer and pt[1] <= (im_size[1]-buffer)

    assert type(going_out) is bool or type(going_out) is np.bool_
    return going_out

def createP (K, c_R_w, w_t_cam):
    '''Returns a 3x4 P matrix, created from a calibration matrix K,
    the world to camera rotation matrix c_R_w, and the location of the camera in the 
    world coordinate system
    '''
    assert_np_matrix(K, (3,3))
    assert_np_matrix(c_R_w, (3,3))
    assert_np_matrix_choice(w_t_cam, [(3,), (3,1), (1,3)] )
    if len(w_t_cam.shape)==2:
        w_t_cam = w_t_cam((3,))
    #Your code goes here...

    assert_np_matrix(going_out, (3,4) )
    return going_out

def projectPoints(P,X):
    '''X is a 3xN or 4xN array with N points in it.  P is the projection matrix.
    Note that this function assumes the 4th element of X (if it is 4xN) are all
    1's.  If not, I don't check and who knows what happens... '''
    assert_np_matrix( P, (3,4) )
    assert_np_matrix_choice(X, [(3,0), (3,)] )
    if len(X.shape) == 1:
        X = X.reshape((3,1))
    
    #Your code goes here

    assert_np_matrix(going_out, (2,0) )
    return going_out

def backprojectPoints(K, c_R_w, w_t_cam, x_dist_tuples):
    '''  
    This function takes in a list of tuples containing an x (image) point and
    a "z" distance.  It then uses K, R, and t to project a point in space.  It
    will return a numpy array of world (3-D) points size 3xN, where N is the 
    length of the input list
    '''
    assert_np_matrix(K, (3,3) )
    assert_np_matrix(c_R_w, (3,3) )
    assert_np_matrix(w_t_cam, (3,))
    for tup in x_dist_tuples:
        assert_np_matrix(tup[0], (2,))
        if not type(tup[1]) is float and not type(tup[1]) is np.float64:
            raise ValueError('Wrong type', 'distance in x_dist_tuples must be floats')

    #Your code goes here...

    assert_np_matrix(going_out, (3,0))
    return going_out

def fundamentalMatrixFromGeometry(K, c1_R_w, w_t_cam1, c2_R_w, w_t_cam2):
    '''
    This function assumes two cameras, both with the same calibration matrix.  
    It should return the fundamental matrix that takes points in camera 1 image space
    and returns a line in camera 2's image space.  The rotation and location
    of the two cameras are with respect to a "world" frame as denoted by the
    parameter notations
    '''
    assert_np_matrix(K, (3,3))
    assert_np_matrix(c1_R_w, (3,3))
    assert_np_matrix(c2_R_w, (3,3))
    assert_np_matrix(w_t_cam1, (3,) )
    assert_np_matrix(w_t_cam2, (3,) )
    #Your code goes here...
    
    assert_np_matrix(going_out, (3,3))
    return going_out

def lineNormalToPoints(n, im_size):
    '''This returns either a tuple or list with two points inside of it.  These
    points should be on the edge of the image and correspond with the normal (n)
    passed in as the first parameter.  i.e. n^T x = 0 where x are the points 
    returned by this function.  The returned points should be 1-d numpy arrays with 
    2 elements.  im_size is assumed to be (width, height)
    '''
    
    assert_np_matrix(n, (3,) )
    assert(len(im_size)==2)

    print("This is a week 3 function")

    #going_out should be a tuple/list with nparray(2,) points it
    for pt in going_out:
        assert_np_matrix(pt, (2,))
    return going_out

def rotateCameraAtPoint( w_point, w_camera, random_roll=False ):
    '''
    This function computes a yaw and pitch to point the center of a camera at
    w_camera at the point at w_point.  w_ means both are in the world coordinate.
    It will return a rotation matrix c_R_w.  If random_roll is true, then a random
    roll will be added to the DCM

    Math-wise, the center image vector ([0,0,1]) gets rotated with yaw (y) and 
    pitch(y) to become:
    [sin(y)cos(p), -sin(p), cos(y)cos(p)]
    '''
    assert_np_matrix(w_point, (3,) )
    assert_np_matrix(w_camera, (3,) )

    #Your code goes here

    assert_np_matrix(going_out, (3,3) )
    return going_out

def backupCamera( w_points, c_R_w, w_camera, K, im_size, pixel_buffer=None ) -> np.ndarray((3,)):
    '''
    This function takes in an array of points in the world coordinate system 
    (w_points, 3xN) and "backs up" the camera location, where the camera is oriented 
    according to c_R_w, until all of the points are in the FoV of the camera.
    The FoV is determined by the K and im_size (width, height) passed in.

    If pixel_buffer is passed in, it will back up the camera further so that no
    pixel is within pixel_buffer pixels of the edge of the image
    '''
    assert_np_matrix(w_points, (3,0) )
    assert_np_matrix(c_R_w, (3,3) )
    assert_np_matrix(w_camera, (3,))
    assert_np_matrix(K, (3,3))
    assert len(im_size)==2

    #Your code goes here...

    assert_np_matrix( going_out, (3,) )
    return going_out

def imPointDerivX(P, X, row=None) -> np.ndarray:
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
    assert_np_matrix(P, (3,4))
    assert_np_matrix(X, (3,))
    
    
    #output is either a 1x3 or 2x3 array, depending on row
    if len(row)==2:
        assert_np_matrix(going_out, (2,3))
    else:
        assert_np_matrix(going_out, (1,3))
    return going_out

def refineTriangulate(Px_list) -> np.ndarray( (3,) ):
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
    num_pts = len(Px_list)
    assert num_pts > 1, "Need at least 2 projection matrices and x values to triangulate"

    for i,PX in enumerate(Px_list):
        P,X = PX
        assert_np_matrix(P, (3,4))
        assert_np_matrix_choice(X, [(2,), (2,1), (1,2)] )
        if len(X.shape)==1 or X.shape == (1,2):
            Px_list[i] = ( P,X.reshape((2,1)) ) #This makes it match the output of projectPoints...


    #Your code goes here...

    #Be careful as the output needs to be (3,), but the return from np.linalg.lstsq (your delta X) is (3,1)
    assert_np_matrix(going_out, (3,))
    return going_out
