import os
import pandas as pd
import numpy as np

def extract_T_matrix_dict(T_matrix_dir,
                          calc_SVD=True,
                          calc_cov=True,
                          calc_inv=True,
                          verbose=0
                         ):
    # create a list of items in T_matrix_dir
    T_matrix_dir_items = os.listdir(T_matrix_dir)

    # Loop through list of paths to process each matrix
    T_matrix_dict = {}
    for T_matrix_name in T_matrix_dir_items[:1]:

        # Create path name
        T_matrix_full_path = os.path.join(T_matrix_dir, T_matrix_name)

        # Extract language strings and extensions
        [T_matrix_lgs, T_matrix_ext] = T_matrix_name.split(".")
        T_matrix_lgs = T_matrix_lgs.rstrip("_T")

        # Filter for non csv extensions
        if T_matrix_ext != "csv":
            if verbose >= 1:
                print("Skipping %s item in %s: extenion is not '.csv'" % (T_matrix_name,T_matrix_dir))

        else:
            # Load in csv as numpy array
            T_matrix = np.loadtxt(open(T_matrix_full_path), delimiter=",")

            ### Add matrices to T_matrix_dict under Lg1-Lg-2 key
            T_matrix_dict[T_matrix_lgs] = {}

            # Raw matrix
            T_matrix_dict[T_matrix_lgs]["T_matrix"] = T_matrix

            # T_cov Covariance matrix TT^{T}
            if calc_cov==True:
                T_cov = np.dot(T_matrix,T_matrix.T)
                T_matrix_dict[T_matrix_lgs]["T_cov"] = T_cov

            # T_inv Matrix Inverse T^{-1}
            if calc_inv==True:
                T_inv = np.linalg.inv(T_matrix)
                T_matrix_dict[T_matrix_lgs]["T_inv"] = T_inv

            # SVD U, \Sigma matrix, and V matrices
            if calc_SVD==True:
                print("have't done SVD coming soon")

            if verbose >= 2:
                print("Matrix %s loaded with shape %r.\nThe following matrices calculated and added to dictionary: %r" % (T_matrix_name, T_matrix.shape,T_matrix_dict[T_matrix_lgs].keys()))
    return T_matrix_dict