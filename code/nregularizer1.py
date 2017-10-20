from __future__ import absolute_import
import six
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object


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


def l1(l=0.1):
    return L1L2(l1=l)


def l2(l=0.1):
    return L1L2(l2=l)

def l3(l=0.1):
    return L1L2(l3=l)


def l1_l2(l1=0.1, l2=0.1):
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
