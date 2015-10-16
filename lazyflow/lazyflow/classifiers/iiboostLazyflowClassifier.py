import cPickle as pickle
from itertools import starmap

import numpy

import logging
logger = logging.getLogger(__name__)

import iiboost

from lazyflow.classifiers import LazyflowPixelwiseClassifierFactoryABC, LazyflowPixelwiseClassifierABC

def roi_to_slice(start, stop):
    return tuple( starmap(slice, zip(start, stop)) )

class IIBoostLazyflowClassifierFactory(LazyflowPixelwiseClassifierFactoryABC):
    """
    This class adheres to the LazyflowPixelwiseClassifierFactoryABC interface, 
    which means it can be used by the standard classifier operators defined in lazyflow.
    
    Instances of this class can create trained instances of IIBoostLazyflowClassifier,
    which adheres to the LazyflowPixelwiseClassifierABC interface.
    
    NOTE: IIBoost needs three different (multi-channel) images:
          - raw data
          - hessian eigenvalues
          - feature channel integral images
          
          To allow us to treat this classifier like a "normal" pixelwise classifier in ilastik/lazyflow,
          all three of images are passed in via the same numpy array.
          By convention, the input array must contain the following channels:
          channel 0: the raw grayscale data
          channel 1-9: the hessian eigenvectors, flattened into 9 channels
          channel 10-N: the remaining feature channels, (as selected by the user), 
                        which must be integral images of the underlying features.
    """
    VERSION = 1
    
    def __init__(self, num_stumps, *args, **kwargs):
        self.num_stumps = num_stumps
        self._args = args
        self._kwargs = kwargs
    
    def create_and_train_pixelwise(self, feature_images, label_images, axistags=None, feature_names=None):
        """
        feature_images: A sequence of ND images.  See note above regarding required structure.  
                        Last axis must be channel.
        label_images: A sequence of ND label images.  Last axis must be channel (size=1).
        axistags: Optional.  A vigra.AxisTags object describing ALL feature_images.
                  (Used to compute the anisotropy of the data.)
        """
        logger.debug( 'training with IIBoost' )

        # Instantiate the classifier
        model = iiboost.Booster()

        # IIBoost requires both labels to be uint8, 3D only
        converted_labels = []
        for label_image in label_images:
            assert len(label_image.shape) == 4, "IIBoost expects 4D data (including channel dimension)."
            assert label_image.shape[-1] == 1, "Expected label image to have only one channel."
            converted = numpy.array( numpy.asarray(label_image[...,0], dtype=numpy.uint8) )
            converted_labels.append( converted )

        # Save for future reference
        flattened_labels = map( numpy.ndarray.flatten, converted_labels )
        all_labels = numpy.concatenate(flattened_labels)
        known_labels = numpy.unique(all_labels)
        if known_labels[0] == 0:
            known_labels = known_labels[1:]
            
        assert set([1,2]).issuperset(known_labels), "IIBoost only accepts two label values: 1 and 2"

        # We can't train if there not labels from both classes.
        if set(known_labels) != set([1,2]):
            return None

        # IIBoost requires raw images to be uint8, 3D only
        # NOTE: we assume that the raw data can be found in channel 0.
        raw_images = []
        for image in feature_images:
            assert len(image.shape) == 4, "IIBoost expects 4D data (including channel)."
            # (Even though copy=False here, we'll probably get a copy anyway. That's fine. Same for cases below, too.)
            raw = image[...,0].astype(dtype=numpy.uint8, order='C', copy=False)
            raw_images.append( raw )

        # Extract the hessian eigenvector (hev) images
        hev_images = []
        for image in feature_images:
            hev_image = image[...,1:10]
            hev_image = hev_image.astype(dtype=numpy.float32, order='C', copy=False)
            hev_image = hev_image.reshape( hev_image.shape[:-1] + (3,3) )
            hev_images.append( hev_image )

        integral_images = []
        for image in feature_images:
            assert len(image.shape) == 4, "IIBoost expects 4D data (including channel dimension)."

            # Select integral channels (see note above about input data)
            integral_image = image[...,10:]
            
            # Put channel first
            integral_image = integral_image.transpose(3,0,1,2)

            # IIBoost requires filter images to be float32, in C-order
            integral_image = integral_image.astype(dtype=numpy.float32, copy=False, order='C')

            integral_images.append( integral_image )
        
        # Calculate anisotropy factor.
        z_anisotropy_factor = 1.0
        if axistags:
            assert [tag.key for tag in axistags] == list('zyxc'), "Data must be provided in zyxc order"
            x_tag = axistags['x']
            z_tag = axistags['z']
            if z_tag.resolution != 0.0 and x_tag.resolution != 0.0:
                z_anisotropy_factor = z_tag.resolution / x_tag.resolution

        model.trainWithChannels( raw_images, 
                                 hev_images, 
                                 converted_labels, 
                                 integral_images, 
                                 z_anisotropy_factor, 
                                 self.num_stumps, 
                                 *self._args, 
                                 **self._kwargs )

        return IIBoostLazyflowClassifier( model, known_labels, feature_count=len(integral_images[0]), feature_names=feature_names )

    def get_halo_shape(self, data_axes):
        # Carlos says a hard-coded halo of 50 voxels should suffice.
        halo_shape = (50,) * (len(data_axes)-1)
        halo_shape += (0,) # no halo for channel
        return halo_shape

    @property
    def description(self):
        return "IIBoost Classifier"

    def __eq__(self, other):
        return (    isinstance(other, type(self))
                and self.num_stumps == other.num_stumps
                and self._args == other._args
                and self._kwargs == other._kwargs )
    def __ne__(self, other):
        return not self.__eq__(other)

# This assertion should pass if lazyflow is available.
from lazyflow.classifiers import LazyflowPixelwiseClassifierFactoryABC
assert issubclass( IIBoostLazyflowClassifierFactory, LazyflowPixelwiseClassifierFactoryABC )

class IIBoostLazyflowClassifier(LazyflowPixelwiseClassifierABC):
    """
    Adapt the IIBoost classifier to the interface lazyflow expects.    
    """
    def __init__(self, model, known_labels, feature_count, feature_names):
        self._known_labels = known_labels
        self._model = model
        self._feature_count = feature_count
        self._feature_names = feature_names
    
    def predict_probabilities_pixelwise(self, input_image, roi, axistags=None):
        """
        feature_image: An ND image.  Last axis must be channel.
        roi: The region of interest (start, stop) within feature_image to predict (e.g. without the halo region)
             Note: roi parameter should not include channel.
                   For example, a valid roi for a zyxc image could be ((0,0,0), (10,20,30))
        axistags: Optional.  A vigra.AxisTags object describing the feature_image.
                  (Used to compute the anisotropy of the data.)
        
        Returns: A multi-channel image (each channel corresponds to a different label class).
                 The result image size is determined by the roi parameter.

        NOTE: See note in factory class above concerning the expected structure of the input image.
        """
        logger.debug( 'predicting with IIBoost' )
        assert len(input_image.shape) == 4, "IIBoost expects 3D data."

        # IIBoost requires raw images to be uint8
        # (Even though copy=False here, we'll probably get a copy anyway.  That's fine.  Same for below.)
        raw = input_image[...,0].astype(dtype=numpy.uint8, order='C', copy=False)

        # Extract hessian eigenvalue channels
        hev_image = input_image[...,1:10].astype(dtype=numpy.float32, order='C', copy=False)
        hev_image = hev_image.reshape( hev_image.shape[:-1] + (3,3) )
        
        # Select integral feature channels 
        #  (see note above about expected input data channels)
        integral_filter_channels = input_image[...,10:]

        # Put channel axis first, so this array looks like a channel-wise list of images
        integral_filter_channels = integral_filter_channels.transpose(3,0,1,2)
        
        # iiboost needs C-contiguous float32 arrays
        integral_filter_channels = integral_filter_channels.astype(dtype=numpy.float32, copy=False, order='C')

        # Calculate anisotropy factor.
        z_anisotropy_factor = 1.0
        if axistags:
            assert [tag.key for tag in axistags] == list('zyxc'), "Data must be provided in zyxc order"
            x_tag = axistags['x']
            z_tag = axistags['z']
            if z_tag.resolution != 0.0 and x_tag.resolution != 0.0:
                z_anisotropy_factor = z_tag.resolution / x_tag.resolution
        
        subROI = iiboost.ROICoordinates()
        subROI.z1, subROI.y1, subROI.x1 = roi[0]
        subROI.z2, subROI.y2, subROI.x2 = roi[1]
        
        prediction_img = self._model.predictWithChannels( raw, 
                                                          hev_image, 
                                                          integral_filter_channels, 
                                                          z_anisotropy_factor, 
                                                          useEarlyStopping=True, 
                                                          subROI=subROI )
        assert prediction_img.dtype == numpy.float32
        assert prediction_img.shape == input_image.shape[:-1], \
            "Output has unexpected shape: Got {}, expected {}"\
            .format( prediction_img.shape, input_image.shape[:-1] )        

        # IIBoost computes predictions only for the subROI, 
        #  but the returned array is as large as the input.
        # Extract the predictied roi now.
        prediction_img = prediction_img[roi_to_slice(*roi)]

        # Convert prediction output to a probability: prob = 1/(1 + exp(-prediction))
        # Optimize: Here we use in-place operations to avoid temporaries. 
        #           For medium-to-large arrays, this is 25-40% faster than using the above formula directly.
        prediction_img *= -1
        numpy.exp(prediction_img, out=prediction_img)
        prediction_img += 1
        numpy.reciprocal(prediction_img, out=prediction_img)
        
        assert prediction_img.min() >= 0.0
        assert prediction_img.max() <= 1.0
        
        # Image from model prediction has no channels,
        #  but lazyflow expects classifiers to produce one channel for each 
        #  label class.  Here, we simply generate the first channel by inverting the previous channel.
        prediction_img_reshaped = numpy.zeros( prediction_img.shape + (2,), dtype=numpy.float32 )
        
        if 1 in self._known_labels:
            prediction_img_reshaped[...,0] = 1.0-prediction_img
        if 2 in self._known_labels:
            prediction_img_reshaped[...,-1] = prediction_img
        
        roi_shape = tuple(numpy.subtract(roi[1], roi[0]))
        assert prediction_img_reshaped.shape == roi_shape + (len(self._known_labels),), \
            "Prediction has wrong shape. Expected: {}, Got {}"\
            "".format( roi_shape + (len(self._known_labels),), prediction_img_reshaped.shape )
        return prediction_img_reshaped
    
    @property
    def known_classes(self):
        return self._known_labels

    @property
    def feature_count(self):
        return self._feature_count

    @property
    def feature_names(self):
        return self._feature_names

    def get_halo_shape(self, data_axes):
        # Carlos says a hard-coded halo of 50 voxels should suffice.
        halo_shape = (50,) * (len(data_axes)-1)
        halo_shape += (0,) # no halo for channel
        return halo_shape

    def serialize_hdf5(self, h5py_group):
        h5py_group['known_labels'] = self._known_labels
        h5py_group['feature_count'] = self._feature_count
        h5py_group['feature_names'] = self._feature_names
        
        # This field is required for all classifiers
        h5py_group['pickled_type'] = pickle.dumps( type(self) )
        
        # Just store the string IIBoost gives us
        h5py_group['serialized_model'] = self._model.serialize()

    @classmethod
    def deserialize_hdf5(cls, h5py_group):
        model_str = h5py_group['serialized_model'][()]
        model = iiboost.Booster()
        model.deserialize(model_str)
                
        known_labels = list(h5py_group['known_labels'][:])
        feature_count = h5py_group['feature_count'][()]
        feature_names = list(h5py_group['feature_names'][:])
        return IIBoostLazyflowClassifier(model, known_labels, feature_count, feature_names)

# This assertion should pass if lazyflow is available.
from lazyflow.classifiers import LazyflowPixelwiseClassifierABC
assert issubclass( IIBoostLazyflowClassifier, LazyflowPixelwiseClassifierABC )
