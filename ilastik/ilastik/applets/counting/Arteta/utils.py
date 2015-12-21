
import os
import errno
from copy import copy, deepcopy

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
from scipy import ndimage

import json

class RandomState(np.random.RandomState):
    
    def __init__(self, seed=None):
        super(RandomState, self).__init__(seed)
        self.seed = seed
    
    def get_params(self):
        return {"seed": self.seed}
    

def grouper(iterable, n):
    return (iterable[i:i+n] for i in xrange(0,len(iterable),n))

def unique_short(name, others):
    """
    Find a short version of ``name`` that is different from the short version of
    the elements in ``others``.
    """
    
    others = set(others)
    others.discard(name)
    
    words = name.split('_')
    words_others = [w.split('_') for w in others]
    
    short = []
    for i, word in enumerate(words):
        
        indices_to_add = {0}
        
        current_words_others = copy(words_others)
        
        for wos in current_words_others:
            if i >= len(wos):
                words_others.remove(wos)
                continue
            
            word_other = wos[i]
            
            if word != word_other:
                index_first_diff = [j for j, (c1, c2) in enumerate(map(None, word, word_other)) if c1 != c2][0]
                words_others.remove(wos)
                indices_to_add.add(index_first_diff)
        
        letters = [word[j] for j in sorted(indices_to_add)]
        short.extend(letters)
    
    short = ''.join(short)
    return short

def short_descriptions(names):
    
    # Avoid repeated names
    names = list(set(names))
    shorts = [unique_short(name, names) for name in names]
    return dict(zip(names, shorts))

def short_keys(d):
    """Returns a dictionary with shortened keys"""
    
    short = short_descriptions(d.keys())
    return {short[key]: d[key] for key in d.keys()}

def dictionary_description(d):
    """Returns a string containing the values in the dictionary"""
    return ','.join(['%s_%s' % (k, d[k]) for k in sorted(d.keys())])

def dictionary_short_description(d, name_key="__class__"):
    """Returns a dictionary description with short keys"""
    name = None
    if d.has_key(name_key):
        d = copy(d)
        name = d[name_key]
        del d[name_key]
    
    d = short_keys(d)
    
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = "(%s)" % dictionary_short_description(v, name_key)
        else:
            res[k] = str(v)
    
    strres = dictionary_description(res)
    if name is None:
        return strres
    
    return "%s_%s" % (name, strres)

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def mklist(lst, dim):
    """
    If *lst* is a scalar value or a sequence with one element, returns a list
    of that element repeated *dim* times.
    
    If *lst* is a sequence of length *dim*, returns *lst*.
    
    Otherwise, it raises a ValueError exception.
    
    Examples:
    >>> mklist(1, 4)
    [1, 1, 1, 1]
    >>> mklist([2], 4)
    [2, 2, 2, 2]
    >>> mklist([1, 2, 3, 4], 4)
    [1, 2, 3, 4]
    >>> mklist([1, 2, 3], 4)
    ValueError: lst has an invalid length
    """
    try:
        d = len(lst)
        if d == 1:
            lst = lst * dim
        elif d != dim:
            raise ValueError("lst has an invalid length")
    except TypeError:
        lst = [lst] * dim
    return lst

def angle(v1, v2):
    
    num = np.dot(v1.ravel(), v2.ravel())
    den = np.sqrt((v1 ** 2).sum()) * np.sqrt((v2 ** 2).sum())
    return num / den

def norm_corr(image, kernel):
    
    num = ndimage.correlate(image, kernel)
    den1 = np.sqrt(ndimage.correlate(image ** 2, np.ones_like(kernel)))
    den2 = np.sqrt((kernel ** 2).sum())
    return num / (den1 * den2)

def load_json(json_file):
    
    with open(json_file, "r") as f:
        config = json.load(f)
    
    return config

def save_json(json_file, config):
    
    with open(json_file, "w") as f:
        json.dump(config, f, indent=4)

def broadcast_to_shape(array, shape):
    
    array = np.asarray(array)
    newshape = tuple(shape)
    
    oldshape = array.shape
    strides = list(array.strides)
    
    if newshape == oldshape:
        return array
    
    diff_dims = len(newshape) - len(oldshape)
    if diff_dims < 0:
        raise ValueError("new shape cannot have less dimensions than the one "
            "of the original array")
    
    oldshape = (1,)*diff_dims + oldshape
    strides = [0,]*diff_dims + strides
    
    for i in range(len(oldshape)):
        
        if oldshape[i] == newshape[i]:
            continue
        
        if oldshape[i] == 1:
            strides[i] = 0
            continue
        
        raise ValueError("incompatible shapes")
    
    broadcasted = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return broadcasted

def block_view(arr, block_shape):
    
    assert arr.ndim == len(block_shape), \
            "ndim mismatch; arr.ndim(=%s) and len(block_shape(=%s) should be equal" % (arr.ndim, len(block_shape))
    assert all([i % j == 0 for i, j in zip(arr.shape, block_shape)]), \
            "block_view requires arr.shape[i] to be a multiple of block_shape[i]"
    
    shape= tuple(i // j for i, j in zip(arr.shape, block_shape)) + block_shape
    strides = tuple(i * j for i, j in zip(arr.strides, block_shape)) + arr.strides
    return ast(arr, shape=shape, strides=strides)

def separate_convolve(image, weights, axis=None):
    
    if axis is None:
        axis = np.arange(image.ndim)
    
    aux = image
    for ax in axis:
        aux = ndimage.convolve1d(aux, weights, axis=ax)
    
    return aux

def non_extrema_suppression(features, size=None, output=None):
    """Set non-extrema to nan"""
    
    # Find maxima
    maxima = ndimage.maximum_filter(features, size) == features
    
    # Find minima
    minima = ndimage.minimum_filter(features, size) == features
    
    extrema = np.logical_xor(maxima, minima)
    
    if output is None:
        output = np.zeros_like(features)
    
    output[extrema] = features[extrema]
    output[np.logical_not(extrema)] = np.nan
    
    return output

def accumulate(lst):
    
    total = 0
    
    for i in lst:
        total += i
        yield total

def split_and_reshape(arr, shapes):
    
    lens = map(np.prod, shapes)
    split_positions = [0] + list(accumulate(lens))
    splitted = [arr[i:j] for i,j in zip(split_positions[:-1], split_positions[1:])]
    
    reshaped = [np.reshape(i, shape) for i, shape in zip(splitted, shapes)]
    return reshaped

def haar_like_1d(size, min_length=1):
    
    max_i = (size // min_length).bit_length() - 1
    indices = np.arange(size, dtype=np.int_)
    
    masks_aux = np.array([(indices // (size//(1<<i))) & 1 != 0 for i in xrange(max_i + 1)])
    masks = np.vstack([masks_aux, ~masks_aux])
    
    return masks

def sample_haar_like_sets(arr, mask, axis, min_length=1):
    
    ndim = mask.ndim
    
    haar_sets = haar_like_1d(arr.shape[axis], min_length=min_length)
    
    res = []
    for haar_set in haar_sets:
        
        if np.count_nonzero(haar_set) == 0:
            continue
        
        slices = tuple(slice(None) if i == axis else None for i in xrange(ndim))
        current_mask = np.logical_and(mask, haar_set[slices])
        
        res.append(arr[current_mask])
    
    return res
