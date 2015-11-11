from lazyflow.graph import Operator, InputSlot, OutputSlot
from functools import partial
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.request import Request
from lazyflow.operators import OpArrayCache, OpBlockedArrayCache

from ilastik.applets.base.applet import DatasetConstraintError
from ilastik.utility import MultiLaneOperatorABC, OperatorSubView

import traceback
import numpy, vigra
import sys
sys.path.append('/Users/Chris/Code/python_tests/slic/') #TODO CHRis - automate this (integrate slic code in Ilastik?)
import slic 

class OpSlic3D(Operator):
    Input = InputSlot()
    # These are the slic parameters.

    SuperPixelSize = InputSlot(value=10)
    Compactness = InputSlot(value=5.0)
    MaxIter = InputSlot(value=6)
    
    Output = OutputSlot(level=1)
    # Boundaries = OutputSlot()


    def SetBoundariesCallback(self, boundaries):
        print 'In SetBoundariesCallback ... '
        assert isinstance(boundaries, numpy.ndarray) , "The returned value to SetBoundariesCallback must be a numpy array"
        print 'Out shape ', self.Output.meta.shape
        print 'Bound shape ', boundaries.shape
        # assert boundaries.shape == self.Output.meta.shape # make sure the shapes match
        # self.Boundaries.value = boundaries[...,None] # sets boundaries


    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Boundaries.meta.assignFrom(self.Input.meta)

        tagged_shape = self.Input.meta.getTaggedShape()
        assert 'c' in tagged_shape, "We assume the image has an explicit channel axis."
        assert tagged_shape.keys()[-1] == 'c', "This code assumes that channel is the last axis."
        
        # Output will have exactly one channel, regardless of input channels
        tagged_shape['c'] = 2
        self.Output.meta.shape = tuple(tagged_shape.values())
        self.Boundaries.meta.shape = tuple(tagged_shape.values())

        # slic.SetPyCallback(self.SetBoundariesCallback) # sets the callback <-- to put somewhere else than setupOutputs?
    
    def execute(self, slot, subindex, roi, result):
        input_data = self.Input(roi.start, roi.stop).wait()
        # boundaries = numpy.ndarray(input_data.shape)
        # numpy.copyto(boundaries, input_data, casting='same_kind', where=None)
        slic_sp = slic.Compute3DSlic(input_data[:,:,:,0], int(self.SuperPixelSize.value),self.Compactness.value,int(self.MaxIter.value))   
        # print 'Data after : ', data
        result[:] = slic_sp[...,None] #<--- Add Channel axis
        
    
    def propagateDirty(self, slot, subindex, roi):
        # For some operators, a dirty in one part of the image only causes changes in nearby regions.
        # But for superpixel operators, changes in one corner can affect results in the opposite corner.
        # Therefore, everything is dirty.
        self.Output.setDirty()

class OpCachedSlic3D(Operator):
    """
    This is the cached operator for the SLIC class. It shouldn't compute anything, 
    just redirect computation to OpSlic and store the results in a cache 
    """
    name = "OpCachedSlic"
    category = "top-level"

    Input = InputSlot() 
    SuperPixelSize = InputSlot(optional=True)
    Cubeness = InputSlot(optional=True)
    MaxIter = InputSlot(optional=True)

    Output = OutputSlot(level=1) # Result of the SLIC algorithm goes in these images
    # Boundaries = OutputSlot()

    def __init__(self, *args, **kwargs):
        super( OpCachedSlic3D, self ).__init__(*args, **kwargs)
        
        # OpSlic to compute SLIC 
        self.opSlic = OpSlic3D(parent=self )
        self.opSlic.Compactness.connect(self.Cubeness)
        self.opSlic.SuperPixelSize.connect(self.SuperPixelSize)
        self.opSlic.MaxIter.connect(self.MaxIter)
        self.opSlic.Input.connect(self.Input) 
        
        self.opSlicCache = OpBlockedArrayCache(parent=self)
        self.opSlicCache.Input.connect(self.opSlic.Output)
        self.Output.connect(self.opSlicCache.Output )
        # self.Boundaries.connect(self.opSlic.Boundaries) # go through cache? shouldn't be necessary

    def setupOutputs(self):        
        # The cache is capable of requesting and storing results in small blocks,
        # but we want to force the entire image to be handled and stored at once.
        # Therefore, we set the 'block shape' to be the entire image -- there will only be one block stored in the cache.
        # (Note: The OpBlockedArrayCache.innerBlockshape slot is deprecated and ignored.)
        self.opSlicCache.outerBlockShape.setValue( self.Input.meta.shape )

        def markAllOutputsDirty( *args ):
            self.propagateDirty( self.Input, (), slice(None) )
        
    
    def execute(self, slot, subindex, roi, result):
        """
        Compute SLIC superpixel segmentation
        """      
        pass # shouldn't do anything, as the computation should be done in OpSlic
        
    def propagateDirty(self, slot, subindex, roi):
        pass # nothing to do here, likewise
     


