from __future__ import absolute_import
import six
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
import pickle
import os
import json
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras import regularizers
from numpy.linalg import det
from numpy.linalg import inv
import bottleneck as bn
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()



class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class L1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0., l3 =0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.l3 = K.cast_to_floatx(l3)

    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        if self.l3:
            regularization += K.sum(self.l3 * K.square(tf.subtract(K.dot(x,K.transpose(x)),K.dot(K.transpose(x),x))))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2),
                'l3': float(self.l3)}


# Aliases.


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)

def l3(l=0.01):
    return L1L2(l3=l)

#Normalizer with regularization

def l3_l2(l3=0.01, l2=0.01):
    return L1L2(l3=l3, l2=l2)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)


def serialize(regularizer):
    return serialize_keras_object(regularizer)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret regularizer identifier:',
                         identifier)


def parse_arguments():
    return argparse.ArgumentParser()

def path_exists(directory):
    return os.path.exists(directory)

def make_dir(directory):
    return os.makedirs(directory)

def pickle_rw(*tuples, write=True):
    """Pickle object in each tuple to/from ../pickle folder
    tuples = the filenames and objects to pickle ('name', name)"""
    result = []
    for tup in tuples:
        fname, obj = tup
        if write:
            with open('../pickle/' + fname + '.pkl', 'wb') as f:
                pickle.dump(obj, f)
        else:
            with open('../pickle/' + fname + '.pkl', 'rb') as f:
                result.append(pickle.load(f, encoding='bytes'))
    if result == []:
        return
    elif len(result) == 1:
        return result[0]
    else:
        return result

def make_df(rows, cols):
    df = pd.DataFrame(0.0, index=rows,columns=cols)
    return df

def convert_mat_to_df(T):
    return pd.DataFrame(T)

def convert_df_to_mat(T):
    return np.matrix(T)

def make_dict(vocab, vectors):
    """Make dictionary of vocab and vectors"""
    return {vocab[i]: vectors[i] for i in range(len(vocab))}


def vocab_train_test(embedding, lg1, lg2, lg1_vocab):
    """Create training and test vocabularies"""
    if embedding == 'zeroshot':
        with open('../data/zeroshot/transmat/data/' +
                  'OPUS_en_it_europarl_train_5K.txt') as f:
            vocab_train = [(_.split(' ')[0], _.split(' ')[1])
                           for _ in f.read().split('\n')[:-1]]
        with open('../data/zeroshot/transmat/data/' +
                  'OPUS_en_it_europarl_test.txt') as f:
            vocab_test = [(_.split(' ')[0], _.split(' ')[1])
                          for _ in f.read().split('\n')[:-1]]

    elif embedding in ['fasttext_random', 'fasttext_top']:
        embedding, split = embedding.split('_')
        lg1_lg2, lg2_lg1 = pickle_rw((lg1 + '_' + lg2, 0),
                                     (lg2 + '_' + lg1, 0), write=False)
        # T = Translation, R = Reverse (translated and then translated back)
        # Create vocab from 2D translations
        vocab_2D = []
        for lg1_word in lg1_vocab:
            # Translate lg1_word
            if lg1_word in lg1_lg2:
                lg1_word_T = lg1_lg2[lg1_word]
                # Check if translated word (or lowercase) is in lg2_lg1
                if lg1_word_T in lg2_lg1.keys():
                    lg1_word_R = lg2_lg1[lg1_word_T]
                elif lg1_word_T.lower() in lg2_lg1.keys():
                    lg1_word_T = lg1_word_T.lower()
                    lg1_word_R = lg2_lg1[lg1_word_T]
                else:
                    lg1_word_R = None

                # Check if lg1_word and lg1_word_R are equal (lowercase)
                if lg1_word_R:
#                     if lg1_word.lower() == lg1_word_R.lower():
                    vocab_2D.append((lg1_word, lg1_word_T))
        print('length of '+ lg1+'-'+ lg2+ ' vocab: '+str(len(vocab_2D)))

        #Create Train/Test vocab

        if split == 'random':
            sample = np.random.choice(len(vocab_2D), 6500, replace=False)
            vocab_train = np.asarray(vocab_2D)[sample[:5000]].tolist()
            vocab_test = np.asarray(vocab_2D)[sample[5000:]].tolist()
        elif split == 'top':
            sample = np.random.choice(range(6500), 6500, replace=False)
            vocab_train = np.asarray(vocab_2D)[:5000, :].tolist()
            vocab_test = np.asarray(vocab_2D)[:1500, :].tolist()
        else:
            pass


    return vocab_train, vocab_test


def vectors_train_test(vocab_train, vocab_test,lg1_dict,lg2_dict):
    """Create training and test vectors"""
    X_train, y_train = zip(*[(lg1_dict[lg1_word], lg2_dict[lg2_word])
                             for lg1_word, lg2_word in vocab_train])
    X_test, y_test = zip(*[(lg1_dict[lg1_word], lg2_dict[lg2_word])
                           for lg1_word, lg2_word in vocab_test])
    return map(np.asarray, (X_train, X_test, y_train, y_test))




def translation_matrix(X_train, y_train, mode):
    """Fit translation matrix T"""

    model = Sequential()
    if mode == "l2":
        model.add(Dense(300, use_bias=False, input_shape=(X_train.shape[1],),kernel_regularizer=l2(0.01)))
    if mode == "l3":
        model.add(Dense(300, use_bias=False, input_shape=(X_train.shape[1],),kernel_regularizer=l3(0.000001)))
    if mode == "l3_l2":
        model.add(Dense(300, use_bias=False, input_shape=(X_train.shape[1],),kernel_regularizer=l3_l2(0.000001,0.01)))

    model.compile(loss='mse', optimizer='adam')
    history = model.fit(X_train, y_train, batch_size=128, epochs=20,
                        verbose=False)
    T = model.get_weights()[0]

    T = np.matrix(T)

    M = np.multiply(np.matrix(T),100)

    T_norm, T_normed = normalize(M)

    I = inv(T)

    Fr_norm = np.linalg.norm(np.matrix(np.subtract(np.matmul(T,T.getH()),np.matmul(T.getH(),T))),'fro')

    if np.array_equal(np.around(np.matmul(T_normed,T_normed.getH())), np.around(np.matmul(T_normed.getH(),T_normed))) == True:
        tf = "True"
    else:
        tf = "False"

    return model, history, T, tf, I, M, Fr_norm



def translation_accuracy(X_test, y_test):
    """Get predicted matrix 'yhat' using 'T' and find translation accuracy"""
    # yhat
    yhat = X.dot(T)
    count = 0
    for i in range(len(y_test)):
        if yhat[i,:].all() == y_test[i,:].all():
            count = count + 1
    accuracy = count/len(y_test)*100
    return accuracy

def SVD(T):
    """Perform SVD on the translation matrix 'T' """
    U, s, Vh = np.linalg.svd(T, full_matrices=False )
    return U, s, Vh

def log(s):
    return np.log10(s)

def T_svd_EDA(s):
    """Perform SVD on the translation matrix 'T' """
    plt.hist(s, bins='auto', range = (0,1),normed = 1)
    plt.show()


def calc_fro(T):
    return np.linalg.norm(np.matrix(np.subtract(np.matmul(T,T.transpose()),np.matmul(T.transpose(),T))),'fro')

def stat_calc(stat, s, fro, acc):
        if stat=='min':
            return min(s)
        elif stat == 'max':
            return max(s)
        elif stat == 'mean':
            return np.mean(s)
        elif stat == 'median':
            return np.median(s)
        elif stat == 'std':
            return np.std(s)
        elif stat == 'fro':
            return fro
        elif stat == 'acc':
            return acc


def normalize(matrix):
    """Normalize the rows of a matrix"""
    matrix_norm = np.linalg.norm(matrix, axis=1)
    matrix_normed = matrix / np.repeat(matrix_norm, matrix.shape[1]). \
        reshape(matrix.shape)
    return matrix_norm, matrix_normed


def translation_results(X, y, vocab, M, lg2_vectors, lg2_vocab):
    """X, y, vocab - The training or test data that you want results for
    T - The translation matrix
    lg2_vectors, lg2_vocab - Foreign language used to find the nearest neighbor
    """

    # Data Prep on Inputs
    X_word, y_word = zip(*vocab)
    X_norm, X_normed = normalize(X)
    y_norm, y_normed = normalize(y)
    lg2_vectors_norm, lg2_vectors_normed = normalize(lg2_vectors)

    # yhat
    yhat = X.dot(M)
    yhat_norm, yhat_normed = normalize(yhat)

    #X_norm = normalize(X)


    # Nearest Neighbors
    neg_cosine = -yhat_normed.dot(lg2_vectors_normed.T)
    ranked_neighbor_indices = bn.argpartition(neg_cosine, 1, axis = 1 )
    # Nearest Neighbor
    nearest_neighbor_indices = ranked_neighbor_indices[:, 0]
    yhat_neighbor = lg2_vectors[nearest_neighbor_indices, :]
    yhat_neighbor_norm, yhat_neighbor_normed = normalize(yhat_neighbor)
    yhat_neighbor_word = np.asarray(lg2_vocab)[nearest_neighbor_indices]

    # Results DF
    cols = ['X_norm', 'y_norm', 'yhat_norm', 'yhat_neighbor_norm',
            'X_word', 'y_word', 'yhat_neighbor_word']
    results_df = pd.DataFrame({'X_norm': X_norm,
                               'y_norm': y_norm,
                               'yhat_norm': yhat_norm,
                               'yhat_neighbor_norm': yhat_neighbor_norm,
                               'X_word': X_word,
                               'y_word': y_word,
                               'yhat_neighbor_word': yhat_neighbor_word,})
    results_df = results_df[cols]
    results_df['neighbor_correct'] = results_df.y_word == \
        results_df.yhat_neighbor_word

    return results_df


def T_norm_EDA(results_df):
    """Plot result norms side-by-side"""
    test_size = results_df.shape[0]
    test_accuracy = round(results_df.neighbor_correct.mean(), 2)

    plot_data = ['X_norm', 'y_norm', 'yhat_norm', 'yhat_neighbor_norm']

    return test_accuracy


def T_pca_EDA(T):
    """PCA on matrix T"""
    T_ss = StandardScaler().fit_transform(T)
    pca = PCA().fit(T_ss)
    n = pca.n_components_


    isotropy = (1 - sum(np.cumsum(pca.explained_variance_ratio_) * 1 / n)) / .5
    return isotropy


def T_report_results(embedding, lg1, lg2, lg1_vectors, lg2_vectors,
                     X_train, X_test, D, results_df, isotropy):
    md = '## ' + lg1.title() + ' to ' + lg2.title() + ' ' + \
        embedding.title() + '  \n'
    md += '- ' + lg1.title() + ' Vocabulary Size = ' + \
        '{:,.0f}'.format(lg1_vectors.shape[0]) + '  \n'
    md += '- ' + lg1.title() + ' Embedding Length = ' + \
        '{:,.0f}'.format(lg1_vectors.shape[1]) + '  \n'
    md += '- ' + lg2.title() + ' Vocabulary Size = ' + \
        '{:,.0f}'.format(lg2_vectors.shape[0]) + '  \n'
    md += '- ' + lg2.title() + ' Embedding Length = ' + \
        '{:,.0f}'.format(lg2_vectors.shape[1]) + '  \n'
    md += '- Train Size = ' + '{:,.0f}'.format(X_train.shape[0]) + '  \n'
    md += '- Test Size = ' + '{:,.0f}'.format(X_test.shape[0]) + '  \n'
    md += '- Determinant = ' + '{:,.0f}'.format(D) + '  \n'

    md += '- <b>Test Accuracy = ' + \
        '{:,.1%}'.format(results_df.neighbor_correct.mean()) + '</b>  \n\n'



    md += '#### Test L2 Norms  \n'
    md += '- X_norm: L2 norms for ' + lg1.title() + ' test vectors  \n'
    md += '- y_norm: L2 norms for ' + lg2.title() + ' test vectors  \n'
    md += '- yhat_norm: L2 norms for X.dot(T) test vectors ' + \
        '(T = translation matrix)  \n'
    md += '- yhat_neighbor norm: L2 norms for nearest neighbor' + \
        'to X.dot(T) in y test vectors  \n'
    md += '![](../images/' + lg1 + '_' + lg2 + '_' + embedding + \
        '_T_norm.png)  \n\n'

    md += '#### Translation Matrix Isotropy  \n'
    md += '- Isotropy = ' + '{:,.1%}'.format(isotropy) + '  \n'
    md += '![](../images/' + lg1 + '_' + lg2 + '_' + embedding + \
        '_T_isotropy.png)  \n\n'

    return md

def create_palette(p_from, p_to, separate, cmap_flag):
    return sns.diverging_palette(p_from, p_to, sep=separate, as_cmap=cmap_flag)

def read_csv(path):
    return pd.read_csv(path)

def load_csv(path):
    return np.loadtxt(path, delimiter=',')

def create_heatmap(df, c_map):
    return sns.heatmap(df,cmap=c_map)

def show_plot():
    return plt.show()


###############################
### Spectral Analysis Tools ###
###############################
def read_json(json_path):
    with open(json_path, "r") as jpath:
        new_dict = json.load(jpath)
    jpath.close()
    return new_dict

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

def ortho_normality(matrix):
    """
    calculates the l2 norm of ||MM^{T}-I||_2
    """
    # Identy matrix
    I = np.identity(matrix.shape[0])

    # calc ortho_norm
    ortho_norm = np.linalg.norm(np.matrix(np.subtract(np.dot(matrix,matrix.T),I)),'fro')
    return ortho_norm

def add_svd_stats(matrix,
                  matrix_dict,
                  stats,
                  calc_SVD=False,
                  log_spectrum=False,
                  normality=False,
                  determinant=False,
                  ortho_norm=False,
                  condition_num=False
                 ):
    """
    Adds selected stats to the T_matrix_dict and returns dict with added stats
    matrix:        numpy array of the matrix to be processed to add to matrix_dict
    matrix_dict:    dictionary with matrix to be added to processing matrix
    stats:          list statistics to store on the spectrum of the T_matrix
    log_spectrum:   Boolean if True (default) will run analysis on the log
                    of the spectrum too.
    """
    # Load the actual matrix
    matrix_dict["matrix"] = matrix

    # Load Normality of the matrix
    if normality==True:
        matrix_dict["normality"] = normality_val(matrix)

    # Load determinant of the matrix
    if determinant==True:
        matrix_dict["determinant"] = np.linalg.det(matrix)

    # Load ortho normality measure of the matrix
    if determinant==True:
        matrix_dict["ortho_norm"] = ortho_normality(matrix)

    # Load the SVD
    if calc_SVD==True:
        # Run SVD and add to sub keys
        U,s,Vh = SVD(matrix)
        matrix_dict["U_rotation"] = U
        matrix_dict["V_rotation_transpose"] = Vh
        matrix_dict["spectral_values"] = s
        # print('matrix_dict["spectral_values"]',list(matrix_dict["spectral_values"]))

        # Conduct spectrum stats analysis
        for stat in stats:
            matrix_dict[stat] = stat_calc(stat, s, fro=None, acc=None)

        # Add condition_num from spectral_values
        if condition_num==True:
            if "max" in stats:
                spec_max = matrix_dict["max"]
                if "min" in stats:
                    spec_min = matrix_dict["min"]

                    matrix_dict["condition_num"] = spec_max/spec_min
                    matrix_dict["log_condition_num"] = log(spec_max)-log(spec_min)
                    print(matrix_dict["condition_num"],matrix_dict["log_condition_num"]) ###***###
                else:
                    print("Cannot calculate 'condition_number' must have 'min' value in stats.")
            else:
                print("Cannot calculate 'condition_number' must have 'max' value in stats.")

        # Add log spectrum
        if log_spectrum==True:
            sl = log(s)
            matrix_dict["log_spectral_values"]= sl

            # Conduct log spectrum stats analysis
            for stat in stats:
                stat_of_log=stat+"_log"
                matrix_dict[stat_of_log] = stat_calc(stat, sl, fro=None, acc=None)

    return matrix_dict


def extract_T_matrix_dict(T_matrix_dir,
                          stats=['min','max','mean','median','std'],
                          calc_cov=True,
                          calc_inv=False,
                          calc_SVD=True,
                          log_spectrum=True,
                          condition_num=True,
                          normality=True,
                          determinant=False,
                          ortho_norm=False,
                          verbose=0
                         ):
    """
    T_matrix_dir:  str location of the dir with the csv files for the translation matrices

    stats:         list of statistics to calculate on spectrum and log specturm if calc_SVD and
                   log_spectrum are set to True respectively

    calc_cov:      Boolean if default True, will process the covariance matrix and spectra in dict with key 'T_cov'
    calc_inv:      Boolean if default True, will process the matrix inverse and spectra in dict with key 'T_inv'

    calc_SVD:      [depricated all values calc SVD stats] Boolean if default True, will process the SVD of the T_matrix, and also calc_cov and calc_inv if
                   they are set to true and add to the dict:

                   U rotation numpy array 2d with key 'U_rotation'
                   V* rotation numpy array 2d with key 'V_rotation_trans'
                   Spectral values list with key 'spectral_values'


    log_spectrum:  Boolean if default True, adds to dict log of the spectral values list with key 'log_spectral_values'
    condition_num: Boolean if default True, the ratio between the largest and smallest singular value
    normality:     Boolean if default True, adds to dict the L2 norm of ||TT* - T*T|| with key 'normality'
    determinant:   Boolean if default False, adds to dict the determinant
    ortho_norm:    Boolean if default False, adds to dict the ||MM^{T}-I||_2
                   where M is the matrix and I is the identity matrix

    """
    # Create a list of items in T_matrix_dir
    T_matrix_dir_items = os.listdir(T_matrix_dir)

    # Loop through list of paths to process each matrix
    T_matrix_dict = {}
    for T_matrix_name in T_matrix_dir_items:

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
            # Initialize matricies to match keys with matrix arrays
            matricies = {}

            # Load in csv as numpy array
            T_matrix = np.loadtxt(open(T_matrix_full_path), delimiter=",")
            matricies["T_matrix"]=T_matrix


            ### Add matrices to T_matrix_dict under Lg1-Lg-2 key
            T_matrix_dict[T_matrix_lgs] = {}

            # Initialize raw T_matrix key
            T_matrix_dict[T_matrix_lgs]["T_matrix"] = {}

            # construct list of matrices to pull SVD stats on
            matrix_keys = ["T_matrix"]
            if calc_cov==True:
                # Add to list of matricies to be processed
                matrix_keys += ["T_cov"]
                # Calculate T_cov Covariance matrix TT^{T}
                T_cov = np.dot(T_matrix,T_matrix.T)
                matricies["T_cov"]=T_cov

                # Initialize T_cov key
                T_matrix_dict[T_matrix_lgs]["T_cov"] = {}

            if calc_inv==True:
                # Add to list of matricies to be processed
                matrix_keys += ["T_inv"]
                # Calculate T_inv Matrix Inverse T^{-1}
                T_inv = np.linalg.inv(T_matrix)
                matricies["T_inv"]=T_inv
                # Initialize T_inv key
                T_matrix_dict[T_matrix_lgs]["T_inv"] = {}


            for matrix_key in matrix_keys:
                # Load T_matrix and spectral analysis stats as configured into T_matrix sub dict
                T_matrix_dict[T_matrix_lgs][matrix_key] = add_svd_stats(matrix=matricies[matrix_key],
                                                                        matrix_dict=T_matrix_dict[T_matrix_lgs][matrix_key],
                                                                        stats=stats,
                                                                        calc_SVD=calc_SVD,
                                                                        log_spectrum=log_spectrum,
                                                                        normality=normality,
                                                                        determinant=determinant,
                                                                        ortho_norm=ortho_norm,
                                                                        condition_num=condition_num
                                                                       )
    return T_matrix_dict

# Removing the matrices if needed
def T_matrix_point_stats(full_dict,
                         out_list=['matrix', 'U_rotation', 'V_rotation_transpose', 'spectral_values', 'log_spectral_values']
                        ):
    """
    removes out_list matrices from 3 levels down in full_dict
    """
    out_dict={}
    for key1 in full_dict.keys():
        out_dict[key1]={}
        for key2 in full_dict[key1].keys():
            out_dict[key1][key2]={}
            for key3 in full_dict[key1][key2].keys():
                if key3 not in out_list:
                    out_dict[key1][key2][key3]=full_dict[key1][key2][key3]
    return out_dict


### Creating the heatmaps ###
def make_heatmap(T_matrix_dict,
                 matrix_type,
                 stat,
                 language_order= ['en','ru','de','es','fr','it', 'zh-CN'],
                 upper_matrix_type= False
                ):
    """
    Makes a heatmap for the specified stat and matrix_type from the T_matrix_dict

    matrix_dict:       dictionary with structure {'lg1_lg2':{"matrix_type":{stats:values}}}
    matrix_type:       string in {'T_matrix', 'T_cov', 'T_inv'} to build the heatmap from
    stat:              specific statistic to populate the heatmap
    language_order:    order of the rows and columns of the heatmap
    upper_matrix_type: Boolean or string, default False
                       If False, all values will be filled in with stats from matrix_type.
                       If str in {'T_matrix', 'T_cov', 'T_inv'} upper matrix will be filled
                       with stats from that matrix type
    """
    # Initialize the heatmap df with 0s
    df = make_df(language_order, language_order)

    # Loop through the language combos (or upper right combos)
    for i in range(len(language_order)):
        for j in range(len(language_order)):
            lg1 = language_order[i]
            lg2 = language_order[j]

            # Make language translation key from languages
            lg_key = str(lg1)+"_"+str(lg2)
            if lg_key not in T_matrix_dict.keys():
                print("Skipping %s: translation stats not available" % lg_key)

            # Pull value
            value = T_matrix_dict[lg_key][matrix_type][stat]

            # Adjust upper vals if configured
            if upper_matrix_type==False:
                upper_value = value
            else:
                upper_value = T_matrix_dict[lg_key][upper_matrix_type][stat]

            # Update value based on upper tri config
            if j<=i:
                df.set_value(lg1,lg2,value)
            else:
                df.set_value(lg1,lg2,upper_value)
    return df

def plot_heatmaps(T_matrix_dict,
                  plotted_stats,
                  display_opt="cols",
                  matrix_types=["T_matrix"],
                  low_c=10,
                  high_c=130,
                  sep_num=8,
                  figuresize=[12,9]
                 ):
    """
    Plots Heatmaps from the T_matrix_dict generated by extract_T_matrix_dict

    T_matrix_dict:  dictionary containing translation matrices, and spectral analysis of them
    plotted_stats:   list of generated point spectral analysis statistics to be plotted
    display_opt:    str in {"cols","num"} if "cols" (default) heatmaps with colors generated
    matrix_types:    List subset of ["T_matrix","T_cov","T_inv"] if "T_inv" only upper triangle
                    is plotted with T_matrix
    low_c:          num, coolest color spectrum number for heatmap
    high_c:         num, warmest color spectrum number for heatmap
    sep_num:        num, number of separators to use for heatmap
    """

    heatmaps={}
    for matrix_type in matrix_types:
        matrix_type_hmaps = {}

        # Initialize matrix_type plotting arguments
        upper_matrix_type = False
        matrix_type_used = matrix_type

        # plot only upper triangle with T_inv
        if matrix_type=="T_inv":
            matrix_type_used = "T_matrix"
            upper_matrix_type = "T_inv"

        for stat in plotted_stats:
            heatmap = make_heatmap(T_matrix_dict=T_matrix_dict,
                                   matrix_type=matrix_type_used,
                                   stat=stat,
                                   language_order= ['en','ru','de','es','fr','it', 'zh-CN'],
                                   upper_matrix_type=upper_matrix_type
                                  )

            # Set title
            if matrix_type=="T_matrix":
                title = "%s of the translation matrix spectrum" % stat

            elif matrix_type=="T_cov":
                title = "%s of the translation covariance matrix spectrum" % stat

            elif matrix_type=="T_inv":
                title = "%s of the translation matrix spectrum with inverse" % stat



            if display_opt=="cols":

                # Adjust colmap direction based on heatmap
                mean_diag = np.diag(heatmap).mean()
                hm_mean = np.array(heatmap).mean()

                if mean_diag>=hm_mean:
                    colmap = sns.diverging_palette(low_c, high_c, sep=sep_num, as_cmap=True)
                else:
                    colmap = sns.diverging_palette(high_c, low_c, sep=sep_num, as_cmap=True)

                # figure sizing
                plt.figure(figsize=(figuresize[0],figuresize[1]))
                hm = sns.heatmap(heatmap,cmap=colmap)
                #sns.plotting_context(font_scale=5)
                #title = hm.set_title(title)

                # Plot
                print(title)
                plt.show()
                hm
            else:
                print(title)
                display(heatmap)
            matrix_type_hmaps[stat] = heatmap

        # Add dict for that matrix type to full heatmap dict
        heatmaps[matrix_type]=matrix_type_hmaps

    return heatmaps
