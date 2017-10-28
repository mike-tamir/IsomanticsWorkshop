from __future__ import absolute_import
import six
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
import pickle
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
    



def pickle_rw(*tuples,write=True):
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




def translation_matrix(X_train, y_train):
    """Fit translation matrix T"""
    
    model = Sequential()
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