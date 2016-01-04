
import logging

import numpy as np
from scipy import ndimage

import utils
import patch_utils
import sampling
from  kdtree import KDTreeTransformer, KDTree

from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

logger = logging.getLogger(__name__)

class ArtetaPipeline(object):
    
    def __init__(self,
                 num_training_samples=None,
                 kernel_size=15,
                 kernel_type='gaussian',
                 avoid_negative_density=False,
                 random_seed=None,
                 maxDepth = 8):
        
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.num_training_samples = num_training_samples
        self.kernel_size = kernel_size
        self.kernel_type = kernel_type
        self.avoid_negative_density = avoid_negative_density
        self.regressor = linear_model.RidgeCV(
                            alphas=[100000.0, 10000.0, 1000.0, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06],
                            cv=None, fit_intercept=False, gcv_mode=None, normalize=False,
                            scoring=None, store_cv_values=True)
        self.maxDepth = maxDepth
    
    # useless function, channels computed by features extractor
    def _compute_channels(self, img, density, mask):
        
        phi = self.fextractor.transform_one(img)
        
        xs = phi[mask]
        ys = density[mask]
        
        return xs, ys
    
    def _compute_histograms(self, img, mask, x=None):

        scaled_x = self.scaler.transform(x)
        leaves_x = self.kdtree.transform(scaled_x)
        num_leaves = self.kdtree.get_output_ndims()
        aux = leaves_x
        res = np.zeros(mask.shape + (num_leaves,), dtype=np.float32)

        # count = np.sum(leaves_x[:])
        # print 'Num leaves : ', num_leaves
        # print 'Scaled_x : ', scaled_x.shape
        # print 'leaves_x : ', leaves_x.shape
        # print 'Sum over scaled x ', count
        # print 'shape res : ', res.shape
        # print 'mask shape : ', mask.shape
        # print 'res[mask] : ', res[mask].shape

        res[mask] = aux
        
        return res
    
    def _extract_training_data(self, img, density, mask):

        coords = sampling.random_coords_from_mask(self.num_training_samples, mask, self.random_state)
        # print 'extract train data ', img.shape, density.shape, mask.shape
        xs, ys = [], []
        for size in [self.kernel_size]:
            if self.kernel_type == 'flat':
                kernel = np.ones(size)
                int_img = utils.separate_convolve(img, kernel, axis=[0, 1, 2])
                int_density = utils.separate_convolve(density, kernel, axis=[0, 1, 2])
            elif self.kernel_type == 'gaussian':
                int_img = ndimage.gaussian_filter(img, sigma=[size, size,  0]) #size, # removed on dimension here to work on 2d images first
                int_density = ndimage.gaussian_filter(density, sigma=[size, size]) #, size
            else:
                raise ValueError, "Unknown kernel_type '%s'" % self.kernel_type
 
            xs.append(int_img[coords])
            ys.append(int_density[coords])
        
        return np.vstack(xs), np.hstack(ys)

    def set_params(self, **args):
        # FIXME : mechanism to deal with missing or unexisting inputs
        self.maxDepth = args['maxDepth']
        self.kernel_size = args['sigma']

    
    def fit(self, imgs, xs,  densities, masks):
        
        # print '[Arteta_pipeline] Fitting Classifier to data...'
        # print imgs.shape, xs.shape, densities.shape, masks.shape

        # only take into account pixels in the boxes
        xs = xs[masks[:,:,0]]

        # Fit the scaler and the kd-tree
        self.scaler = StandardScaler()
        scaled_xs = self.scaler.fit_transform(np.vstack(xs))

        # print 'Shape of features before/after scaling',np.vstack(xs).shape, scaled_xs.shape
        self.kdtree = KDTreeTransformer(self.maxDepth)
        self.kdtree.fit(scaled_xs)
        
        # Generate histograms
        histograms = self._compute_histograms(imgs,masks,xs) # map(self._compute_histograms, imgs, masks, xs) #
        
        xs, ys = zip(*map(self._extract_training_data, histograms, densities, masks)) #self._extract_training_data(histograms, densities, masks)) #
        xs = np.vstack(xs)
        ys = np.hstack(ys)
        
        # Fit regressor
        self.regressor.fit(xs, ys)
        
        indices = np.arange(xs.shape[1])
        if self.avoid_negative_density:
            
            while np.any(self.regressor.coef_ < 0):
                indices = indices[np.flatnonzero(self.regressor.coef_ > 0)]
                self.regressor.fit(xs[:, indices], ys)
        
        self.coef_ = np.zeros(xs.shape[1], dtype=np.float_)
        self.coef_[indices] = self.regressor.coef_
        self.intercept_ = self.regressor.intercept_
        
        return self
    
    def predict_one(self, img, mask):

        logger.info("Computing histograms...")
        histograms = self._compute_histograms(None, mask,np.vstack(img))
        logger.info("Predicting densities...")
        pred = np.dot(histograms[mask], self.coef_) + self.intercept_
        
        res = np.zeros(img.shape[:-1], dtype=np.float32)
        res[mask[:,:,0]] = pred
        # print 'Results shape : ', res.shape
        # print '[Arteta_pipeline] - Predict one - Sum of predictions is : ', np.sum(res[...])
        return res
    
    def predict(self, imgs, masks):
        # print 'predicting for : ', imgs.shape, masks.shape
        
        return map(self.predict_one, imgs, masks)
    
    def get_params(self):
        return {#"feature_extractor": self.fextractor,
                "regressor": self.regressor,
                "num_training_samples": self.num_training_samples,
                "kernel_size": self.kernel_size,
                "kernel_type": self.kernel_type,
                "random_seed": self.random_seed,
                "avoid_negative_density": self.avoid_negative_density}
    
