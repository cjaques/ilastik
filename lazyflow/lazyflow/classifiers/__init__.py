from lazyflowClassifier import LazyflowVectorwiseClassifierABC, LazyflowVectorwiseClassifierFactoryABC, LazyflowPixelwiseClassifierABC, LazyflowPixelwiseClassifierFactoryABC
from vigraRfLazyflowClassifier import VigraRfLazyflowClassifier, VigraRfLazyflowClassifierFactory
from parallelVigraRfLazyflowClassifier import ParallelVigraRfLazyflowClassifier, ParallelVigraRfLazyflowClassifierFactory
from sklearnLazyflowClassifier import SklearnLazyflowClassifier, SklearnLazyflowClassifierFactory

# Testing
from vigraRfPixelwiseClassifier import VigraRfPixelwiseClassifier, VigraRfPixelwiseClassifierFactory

# IIBoost
try:
    from iiboostLazyflowClassifier import IIBoostLazyflowClassifier, IIBoostLazyflowClassifierFactory
except ImportError:
    import warnings
    warnings.warn("Couldn't import IIBoost classifier.")
    