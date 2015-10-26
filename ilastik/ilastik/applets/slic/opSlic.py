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

class OpSlic(Operator):
    Input = InputSlot()
    # These are the slic parameters.
    # Here we give default values, but they can be changed.
    SuperPixelSize = InputSlot(value=10)
    Compactness = InputSlot(value=5.0)
    MaxIter = InputSlot(value=6)
    
    Output = OutputSlot()
    
    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)

        tagged_shape = self.Input.meta.getTaggedShape()
        assert 'c' in tagged_shape, "We assume the image has an explicit channel axis."
        assert tagged_shape.keys()[-1] == 'c', "This code assumes that channel is the LAST axis."
        
        # Output will have exactly one channel, regardless of input channels
        tagged_shape['c'] = 1
        self.Output.meta.shape = tuple(tagged_shape.values())
    
    def execute(self, slot, subindex, roi, result):
        input_data = self.Input(roi.start, roi.stop).wait()
        slic_sp = slic.Compute2DSlic(input_data,int(self.SuperPixelSize.value), self.Compactness.value, int(self.MaxIter.value));
        result[:] = slic_sp #[...,None] <--- Stuart added a None axis because his SLIC implementation returned a 2D+1 array, without C channel. 
                                            # Ours returns the same size as input, thus 2D+1 with C channel.
    
    def propagateDirty(self, slot, subindex, roi):
        # For some operators, a dirty in one part of the image only causes changes in nearby regions.
        # But for superpixel operators, changes in one corner can affect results in the opposite corner.
        # Therefore, everything is dirty.
        self.Output.setDirty()

class OpCachedSlic(Operator):
    """
    This is the cached operator for the SLIC class. It shouldn't compute anything, 
    just redirect computation to OpSlic and store the results in a cache 
    """
    name = "OpCachedSlic"
    category = "top-level"

    Input = InputSlot() # level = 1 so that input can be multiple images 
    SuperPixelSize = InputSlot(optional=True)
    Cubeness = InputSlot(optional=True)
    MaxIter = InputSlot(optional=True)
    Output = OutputSlot() # Result of the SLIC algorithm goes in these images
    
    def __init__(self, *args, **kwargs):
        super( OpCachedSlic, self ).__init__(*args, **kwargs)
        
        # OpSlic to compute SLIC 
        self.opSlic = OpSlic(parent=self )
        self.opSlic.Compactness.connect(self.Cubeness)
        self.opSlic.SuperPixelSize.connect(self.SuperPixelSize)
        self.opSlic.MaxIter.connect(self.MaxIter)
        self.opSlic.Input.connect(self.Input) 
        
        # OpSlicCache to cache results (operator Wrapper to promote OpSlicCache.input to level=1) 
        self.opSlicCache = OpBlockedArrayCache(parent=self)
        self.opSlicCache.Input.connect(self.opSlic.Output)
        self.Output.connect(self.opSlicCache.Output )

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
     


