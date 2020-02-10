import VANlib as van
import math as m
import numpy as np
import scipy.linalg as la
import unittest

class TestVANlib(unittest.TestCase):
    '''
    ###########################################################
    # These unit tests are for a createRot in the NED coordinate system.  createRot
    is now in a camera (EDN) coordinate system, so these are not really valid anymore
    #####################
    def test_createRot_yaw_degrees(self):
        val = 1/m.sqrt(2.)
        M = van.createRot((0., 0., 45.), True)
        self.assertTrue(np.allclose(M, np.array([[val, val, 0],[-val, val, 0.],[0., 0., 1.]])))

    def test_createRot_yaw_radians(self):
        val = 1/m.sqrt(2.)
        self.assertTrue(np.allclose(van.createRot((0.,0.,m.radians(-45.))),np.array([[val, -val, 0],[val, val, 0.],[0., 0., 1.]])))

    def test_createRot_pitch_radians(self):
        val = 1/m.sqrt(2.)
        self.assertTrue(np.allclose(van.createRot((0.,m.radians(45.),0.)),np.array([[val, 0., -val],[0., 1., 0.],[val, 0., val]])))

    def test_createRot_pitch_degrees(self):
        val = 1/m.sqrt(2.)
        self.assertTrue(np.allclose(van.createRot((0.,-45.,0.), True),np.array([[val, 0., val],[0., 1., 0.],[-val, 0., val]])))

    def test_createRot_roll_radians(self):
        val = 1/m.sqrt(2.)
        self.assertTrue(np.allclose(van.createRot((m.radians(45.),0.,0.)),np.array([[1.0, 0., 0.],[0., val, val],[0., -val, val]])))

    def test_createRot_roll_degrees(self):
        val = 1/m.sqrt(2.)
        self.assertTrue(np.allclose(van.createRot((-45.,0.,0.), True),np.array([[1.0, 0., 0.],[0., val, -val],[0., val, val]])))

    def test_createRot_yaw_pitch_order(self):
        pt = np.array([20., -20., 0.])
        M = van.createRot((0., 32., 45.),True) # The 32 doesn't matter, just non-zero
        pt2 = np.dot(M,pt)
        test = np.array([0, -20*m.sqrt(2.), 0.])
        self.assertTrue(np.allclose(pt2, test))

    def test_createRot_yaw_roll_order(self):
        pt = np.array([10., 10., 0.])
        M = van.createRot((-15, 0., 45.),True) # The -15 doesn't matter, just non-zero
        pt2 = np.dot(M,pt)
        test = np.array([10.*m.sqrt(2), 0., 0.])
        self.assertTrue(np.allclose(pt2, test))

    def test_createRot_pitch_roll_order(self):
        pt = np.array([15., 0., 15.])
        M = van.createRot((7., -45., 0.),True) # The 7 doesn't matter, just non-zero
        pt2 = np.dot(M,pt)
        test = np.array([15*m.sqrt(2), 0., 0.])
        print('pt2 is',pt2,'pt is ',pt)
        self.assertTrue(np.allclose(pt2, test))

    #### end invalidated due to different axes...
    '''

    #Used for several tests.  Rather than re-defining everywhere, just use once.
    K = np.array([[700., 0., 320.],
                [0., 700., 240.],
                [0., 0.,   1.]])

    def test_projectPoints(self):
        #Really, this tests createP as well...
        X_test = np.array([[0., 1., -1., 0., 0.],[0., 0., 0., 1., -1.], [10., 10., 20., 10., 20.]])
        K_test = np.array([[1000., 0., 500.],[0., 1000., 500.], [0., 0., 1.]])
        R1 = np.eye(3)
        t1 = np.zeros((3,))
        x1 = van.projectPoints( van.createP( K_test, R1, t1), X_test )
        should_be1 = 500. + np.array([[0., 100., -50., 0., 0.],[0., 0., 0., 100., -50.]])
        self.assertTrue(np.allclose(should_be1, x1))
        
        #Shift the camera right 1 meter
        t2 = np.array([1., 0., 0.]).reshape((3,))
        x2 = van.projectPoints( van.createP( K_test, R1, t2), X_test )
        should_be2 = 500. + np.array([[-100., 0, -100., -100., -50.],[0., 0., 0., 100., -50.]])
        self.assertTrue(np.allclose(should_be2, x2))

        #Shift the camera down 1 meter
        t3 = np.array([0., 1., 0.]).reshape((3,))
        x3 = van.projectPoints( van.createP( K_test, R1, t3), X_test )
        should_be3 = 500. + np.array([[0., 100., -50., 0., 0.],[-100., -100., -50., 0., -100.]])
        self.assertTrue(np.allclose(should_be3, x3))

        #Now to test rotation somehow...
        R2 = van.createRot((1., 2., 3.), True)
        R_X_test = np.dot(R2.T,X_test)
        x4 = van.projectPoints( van.createP( K_test, R2, t1), R_X_test)
        self.assertTrue(np.allclose(should_be1, x4))

    def createRandomWorldPoints(self, K, c_R_w, w_t_cam, num_pts, dist_range = None):
        '''Create num_pts random world locations within the view of the camera
        at location w_t_cam and with orientation c_R_w.  dist_range defines how
        far away from the camera the points can be.  Default is between 1 and 51 meters.
        
        Note that "in camera field of view" is assumed to be at pixel locations between
        0 and 2*the center offset location in K.  i.e. x is between 0 and 2*K[0,2] and 
        y is between 0 and 2*K[1,2]

        This function returns a tuple of the world points and their pixel locations
        '''
        if dist_range is None:
            dist_range = (1.,51.)
        diff_dist = dist_range[1] - dist_range[0]

        d_rand = np.random.rand(num_pts)*diff_dist + dist_range[0]
        #Create some random image locations, assume a 640x480 image...
        x_rand = np.random.rand(2,num_pts)
        x_rand[0,:] *= 2*K[0,2]
        x_rand[1,:] *= 2*K[1,2]
        x_d_tuples = list((x_rand[:,ii],d_rand[ii]) for ii in range(num_pts))
        #Took random image points and projected them into space
        return (van.backprojectPoints(self.K, c_R_w, w_t_cam, x_d_tuples), x_rand)
        
    def test_projectPoints_and_backProject(self):
        #This tests that backProject is the opposite of projectPoints.  They could both
        #be wrong in the same way and this would not discover it.  Therefore a unit
        #test for one or the other should be used.  This should then confirm that they
        #are both okay (at least that is my thinking)
        n_random_locs = 10
        n_random_pts = 10

        for ii in range(n_random_locs):
            #Create a random rotation and location
            rand_rot = np.random.rand(3,1) * 2*m.pi - m.pi
            #skew symmetric
            SS = van.createSkewSymmMat( rand_rot )
            R_rand = la.expm(SS)
            #Somewhere in a 400x400x400 cube centered on the origin
            t_rand = np.random.rand(3) * 400 - 200
            #Generate the world points
            X,x_rand = self.createRandomWorldPoints(self.K, R_rand, t_rand, n_random_pts)
            #Now take those random world points and project them back into the image
            x_after = van.projectPoints(van.createP( self.K, R_rand, t_rand ), X)
            self.assertTrue(np.allclose(x_after, x_rand))

    def test_backupCamera(self):
        #I will create a bunch of points in front of a camera, then move it forward.
        #After that, call backupCamera and make sure it ends up equal to or forward of
        #the original location
        rotation = np.zeros((3))
        rotation[0] = np.random.rand(1) * m.pi * 2 - m.pi
        R = van.createRot(rotation)
        im_size = (self.K[0,2]*2., self.K[1,2]*2.)
        pts = self.createRandomWorldPoints( self.K, R, np.zeros((3)), 10, dist_range = (10,50))[0]
        new_loc = np.array([0., 0., 9.])
        bu_loc = van.backupCamera(pts, R, new_loc, self.K, im_size)
        self.assertTrue( bu_loc[0] == 0.)
        self.assertTrue( bu_loc[1] == 0.)
        self.assertTrue( bu_loc[2] >= 0.)
        #Now test the projections...
        P = van.createP(self.K, R, bu_loc)
        im_pts = van.projectPoints(P,pts)
        for i in range(im_pts.shape[1]):
            self.assertTrue( van.inImage(im_pts[:,i], im_size))
        
        #Now do the same thing, but with a buffer...
        bu_loc = van.backupCamera(pts, R, new_loc, self.K, im_size, 10)
        #Now test the projections...
        P = van.createP(self.K, R, bu_loc)
        im_pts = van.projectPoints(P,pts)
        for i in range(im_pts.shape[1]):
            self.assertTrue( van.inImage(im_pts[:,i], im_size, 10 ) )
        
        #Might be nice to do a more complex test (rotation / translation), but 
        # this is at least a start
        #TODO:  More complex test here?
        #Note, the TODO is for me, the instructor.  Not you (the student)
    
    def test_rotateCameraAtPoint(self):
        N = 20 #number of times to test it...

        for i in range(N):
            #create two random points, one for the camera, one for the point
            w_cam = np.random.rand(3)*200-100
            w_pt = np.random.rand(3)*200-100
            R = van.rotateCameraAtPoint( w_pt, w_cam, True )
            tst = van.projectPoints( van.createP(self.K, R, w_cam), w_pt ).reshape((2))
            #Point should be at cx, cy in K matrix
            self.assertTrue( np.allclose( tst, self.K[:2,2] ) )
        
    
    def test_imPointDerivX(self):
        #This will be numerical test
        #Create a random world location and some random points in space
        #Move the points by a small amount (numerical derivative)
        #Compare with computed derivative
        N_tests = 20
        ss = .01 # step size
        for i in range(N_tests):
            #create random im_size
            im_size = np.random.rand(2) * 2000. +400. #between 400 and 2400
            focal_length = np.random.rand(1).item() * 2000. + 500.
            K = np.array([[focal_length, 0., im_size[0]/2.0],
                          [0., focal_length, im_size[1]/2.0],
                          [0., 0., 1.]])
            rot = np.random.rand(3)*2*m.pi
            R = van.createRot(rot)
            w_cam = np.random.rand(3) * 400. - 200.
            P = van.createP(K, R, w_cam)
            true_x_loc = np.random.rand(2) * im_size
            X = van.backprojectPoints( K, R, w_cam, [( true_x_loc, ( np.random.rand(1)*500+50).item() )] ).reshape((3,))
            X += np.random.rand(3)*2 -1
            im_x = van.projectPoints(P, X).reshape((2,))
            #This function should do a "closed form" derivative
            comput_deriv = van.imPointDerivX(P, X)
            #Compute a numerical derivative
            numer_deriv = np.zeros((2,3))
            for j in range(3):
                curr_X = X.copy()
                curr_X[j] += ss
                curr_im_pt= van.projectPoints(P, curr_X).reshape((2,))
                numer_deriv[:,j] = (curr_im_pt - im_x)/ss
            #Compare the numerical and closed form derivative
            self.assertTrue(np.allclose(comput_deriv, numer_deriv, rtol=ss*.1))
        
        #I should really do a test to make sure that if row == 0 or row==1 it doesn't croak
        comput_deriv = van.imPointDerivX(P,X)
        comput_deriv0 = van.imPointDerivX(P,X,0).reshape((1,3))
        comput_deriv1 = van.imPointDerivX(P,X,1).reshape((1,3))
        test_deriv = np.concatenate((comput_deriv0, comput_deriv1),axis=0)
        self.assertTrue( np.allclose( comput_deriv, test_deriv ) )

    def test_refineTriangulate(self):
        #First, create the true point
        X_true = np.random.rand(3)*400-200.
        #Find the location of the object in 4 cameras
        N_cams = 4
        w_cams = np.zeros((N_cams,3))
        Rs = []
        Ks = []
        Ps = []
        x_true=[]
        for i in range(N_cams):
            w_cams[i] = np.random.rand(3)*400-200
            Rs.append(van.rotateCameraAtPoint(X_true, w_cams[i], True))
            im_size = np.random.rand(2) * 2000. +400. #between 400 and 2400
            focal_length = np.random.rand(1).item() * 2000. + 500.
            Ks.append(self.K)
            '''np.array([[focal_length, 0., im_size[0]/2.0],
                          [0., focal_length, im_size[1]/2.0],
                          [0., 0., 1.]]))'''
            Ps.append(van.createP(Ks[i], Rs[i], w_cams[i]))
            x_true.append(van.projectPoints(Ps[i], X_true))
        PX_list = list(zip(Ps,x_true))
        X_ret = van.refineTriangulate(PX_list)
        #I choose the atol to correspond with the convergence criteria 
        # in refineTriangulate
        self.assertTrue(np.allclose( X_ret, X_true, atol=1E-5))

unittest.main()

