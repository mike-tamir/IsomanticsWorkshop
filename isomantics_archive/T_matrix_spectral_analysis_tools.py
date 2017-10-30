from __future__ import absolute_import
import os
import pandas as pd
import numpy as np

from ismtools import SVD, log, stat_calc, make_df

def normality_val(matrix):
    """
    matrix: numpy 2d array
    RETURNS
    A measure of the "closeness to normality" of matrix as follows: ||M*M - MM*||_2
    where the norm is taken as the L2 norm for matrices and M is the matrix
    Normal matrices should score 0.
    """
    T = matrix
    T_trans = matrix.transpose()
    matrix_normality = np.linalg.norm(np.matrix(np.subtract(np.matmul(T,T_trans),np.matmul(T_trans,T))),'fro')
    return matrix_normality


def add_svd_stats(matrix_dict, matrix_key, T_matrix_lgs, stats, log_spectrum=False):
    """
    Adds selected stats to the T_matrix_dict and returns dict with added stats
    
    matrix_dict:    dictionary with matrix to be processed and under the T_matrix_lgs key
                    structure of the dictionary {T_matrix_lgs:{matrix_key:{stat: value}}}
                    
    T_matrix_lgs:   str l1_l2 of the two languages
    matrix_key:     str of the matrix to be processed in matrix_dict
    stats:          list statistics to store on the spectrum of the Tmatrix
    log_spectrum:   Boolean if True (default) will run analysis on the log 
                    of the spectrum too.
    """
    # pull the matrix using provided keys
    T_matrix_dict = matrix_dict
    matrix = T_matrix_dict[T_matrix_lgs][matrix_key]
    
    # Run SVD and add to sub keys
    U,s,Vh = SVD(matrix)
    T_matrix_dict[T_matrix_lgs][matrix_key]["U_rotation"] = U
    T_matrix_dict[T_matrix_lgs][matrix_key]["V_rotation_transpose"] = Vh
    T_matrix_dict[T_matrix_lgs][matrix_key]["spectral_values"] = s

    # Normality of T_matrix
    if normality==True:
        T_matrix_dict[T_matrix_lgs][matrix_key]["normality"] = normality_val(matrix)
    
    # Conduct spectrum stats analysis
    for stat in stats:
        T_matrix_dict[T_matrix_lgs][matrix_key][stat] = stat_calc(stat, s, fro=None, acc=None)


    # Add log spectrum
    if log_spectrum==True:
        sl = log(s)
        T_matrix_dict[T_matrix_lgs][matrix_key]["log_spectral_values"]= sl

        # Conduct log spectrum stats analysis
        for stat in stats:
            stat_of_log=stat+"_log"
            T_matrix_dict[T_matrix_lgs][matrix_key][stat_of_log] = stat_calc(stat, sl, fro=None, acc=None)
    
    return T_matrix_dict


def extract_T_matrix_dict(T_matrix_dir,
                          stats=['min','max','mean','median','std'],
                          calc_cov=True,
                          calc_inv=True,
                          calc_SVD=True,
                          
                          log_spectrum=True,
                          normality=True,
                          verbose=0
                         ):
    """
    T_matrix_dir:  str location of the dir with the csv files for the translation matrices
    
    stats:         list of statistics to calculate on spectrum and log specturm if calc_SVD and 
                   log_spectrum are set to True respectively
                   
    calc_cov:      Boolean if default True, will process the covariance matrix and spectra in dict with key 'T_cov'
    calc_inv:      Boolean if default True, will process the matrix inverse and spectra in dict with key 'T_inv'

    calc_SVD:      Boolean if default True, will process the SVD of the T_matrix, and also calc_cov and calc_inv if 
                   they are set to true and add to the dict:
                   
                   U rotation numpy array 2d with key 'U_rotation'    
                   V* rotation numpy array 2d with key 'V_rotation_trans' 
                   Spectral values list with key 'spectral_values'

    
    log_spectrum:  Boolean if default True, adds to dict log of the spectral values list with key 'log_spectral_values'
    normality:     Boolean if default True, adds to dict the L2 norm of ||TT* - T*T|| with key 'normality'
    """
    # create a list of items in T_matrix_dir
    T_matrix_dir_items = os.listdir(T_matrix_dir)

    # Loop through list of paths to process each matrix
    T_matrix_dict = {}
    for T_matrix_name in T_matrix_dir_items[2:3]:

        # Create path name
        T_matrix_full_path = os.path.join(T_matrix_dir, T_matrix_name)

        # Extract language strings and extensions
        [T_matrix_lgs, T_matrix_ext] = T_matrix_name.split(".")
        T_matrix_lgs = T_matrix_lgs.rstrip("_T")

        # Filter for non csv extensions
        if T_matrix_ext != "csv":
            if verbose >= 1:
                print("Skipping %s item in %s: extension is not '.csv'" % (T_matrix_name,T_matrix_dir))

        else:
            # Load in csv as numpy array
            T_matrix = np.loadtxt(open(T_matrix_full_path), delimiter=",")

            ### Add matrices to T_matrix_dict under Lg1-Lg-2 key
            T_matrix_dict[T_matrix_lgs] = {}

            # Raw matrix
            T_matrix_dict[T_matrix_lgs]["matrix"]["T_matrix"] = T_matrix

            # SVD U, \Sigma matrix (s), and Vh matrices
            if calc_SVD==True:
                U,s,Vh = SVD(T_matrix)
                T_matrix_dict[T_matrix_lgs]["U_rotation"] = U
                T_matrix_dict[T_matrix_lgs]["V_rotation_trans"] = Vh
                T_matrix_dict[T_matrix_lgs]["spectral_values"] = s

                # Conduct spectrum stats analysis
                for stat in stats:
                    T_matrix_dict[T_matrix_lgs][stat] = stat_calc(stat, s, fro=None, acc=None)

                
                # Add log spectrum
                if log_spectrum==True:
                    sl = log(s)
                    T_matrix_dict[T_matrix_lgs]["log_spectral_values"]= sl

                    # Conduct log spectrum stats analysis
                    for stat in stats:
                        T_matrix_dict[T_matrix_lgs][stat] = stat_calc(stat, sl, fro=None, acc=None)

                    
                    
            # T_cov Covariance matrix TT^{T}
            if calc_cov==True:
                T_cov = np.dot(T_matrix,T_matrix.T)
                T_matrix_dict[T_matrix_lgs]["matrix"]["T_cov"] = T_cov

            # T_inv Matrix Inverse T^{-1}
            if calc_inv==True:
                T_inv = np.linalg.inv(T_matrix)
                T_matrix_dict[T_matrix_lgs]["matrix"]["T_inv"] = T_inv
            
            
            

            if verbose >= 2:
                print("Matrix %s loaded with shape %r.\nThe following matrices calculated and added to dictionary: %r" % (T_matrix_name, T_matrix.shape,T_matrix_dict[T_matrix_lgs].keys()))
    return T_matrix_dict


