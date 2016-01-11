from lazyflow.graph import Operator, InputSlot, OutputSlot
from functools import partial
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.request import Request
from lazyflow.operators import OpArrayCache, OpBlockedArrayCache

from ilastik.applets.base.applet import DatasetConstraintError
from ilastik.utility import MultiLaneOperatorABC, OperatorSubView

import skimage.segmentation

import sys, traceback, time
import numpy, vigra
import slic 

class OpSlic3D(Operator):
    Input = InputSlot()

    # Slic parameters.
    SuperPixelSize = InputSlot(value=10)
    Compactness = InputSlot(value=5.0)
    MaxIter = InputSlot(value=6)
    
    Output = OutputSlot()
    Boundaries = OutputSlot()
    tempArray = numpy.array((10,10,10)) #tuple()) 

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.Input.meta)
        self.Boundaries.meta.assignFrom(self.Input.meta)
        
        tagged_shape = self.Input.meta.getTaggedShape()
        assert 'c' in tagged_shape, "We assume the image has an explicit channel axis."
        assert tagged_shape.keys()[-1] == 'c', "This code assumes that channel is the last axis."
        
        # Output will have exactly one channel, regardless of input channels
        tagged_shape['c'] = 1
        self.Output.meta.shape = tuple(tagged_shape.values())
        self.Boundaries.meta.shape = tuple(tagged_shape.values())

        slic.SetPyCallback(self.SetBoundariesCallback) # sets the callback <-- to put somewhere else than setupOutputs?
    
    def execute(self, slot, subindex, roi, result):
        
        input_data = self.Input(roi.start, roi.stop).wait()

        if(slot is self.Output):
            if len(input_data.shape) == 3 :
                slic_sp =    slic.Compute2DSlic(input_data, 
                                                int(self.SuperPixelSize.value),
                                                self.Compactness.value,
                                                int(self.MaxIter.value))   
                result[:] = slic_sp
            elif len(input_data.shape) == 4 :
                # start_time = time.time()
                slic_sp =    slic.Compute3DSlic(input_data[:,:,:,0], 
                                                int(self.SuperPixelSize.value), 
                                                self.Compactness.value, 
                                                int(self.MaxIter.value))   
                # print("--- %s seconds ---" % (time.time() - start_time))
                # start_time = time.time()
                # slic_sp = skimage.segmentation.slic(input_data,
                #                             n_segments=638,
                #                             compactness=self.Compactness.value)  
                # print("--- %s seconds ---" % (time.time() - start_time))

                result[:] = slic_sp[...,None] #<--- Add Channel axis
            else:
                assert False, "Can't be here, dimensions of input array have to match 2D or 3D "
        elif slot is self.Boundaries:
            # print 'Updating self.Boundaries'
            # print 'Slot status : ', slot.ready()
            # traceback.print_stack()
            # print '-------------------------------------'
            if len(self.tempArray.shape) > 3 and (self.tempArray.shape == input_data.shape):  # this is necessary when processing a new volume
                result[:] = self.tempArray[:,:,:]
            else:
                # if shapes don't match (when new volume is processed), return input.
                # this layer will be updated later thanks to SetBoundariesCallback
                result[:] = input_data[:]

        else: # for futur layers
            print 'OpSlic3D : returning default, layer ', slot, ' not implemented yet'
            result[:]  = input_data[:]
        
    
    def propagateDirty(self, slot, subindex, roi):
        # For some operators, a dirty in one part of the image only causes changes in nearby regions.
        # But for superpixel operators, changes in one corner can affect results in the opposite corner.
        # Therefore, everything is dirty.
        self.Boundaries.setDirty()
        self.Output.setDirty()

    def SetBoundariesCallback(self, boundaries):
        assert isinstance(boundaries, numpy.ndarray) , "The returned value to SetBoundariesCallback must be a numpy array"
        self.tempArray.resize(boundaries[:,:,:,None].shape)
        self.tempArray[:] = boundaries[:,:,:,None] 
        # Updated self.Boundaries, update display with setDirty.
        self.Boundaries.setDirty() # this doesn't cause the display to be updated on the first time...why?

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

    Output = OutputSlot() # Result of the SLIC algorithm goes in these images
    Boundaries = OutputSlot() # Boudaries between labels

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

        self.opSlicBoundariesCache = OpBlockedArrayCache(parent=self)
        self.opSlicBoundariesCache.Input.connect(self.opSlic.Boundaries)
        self.Boundaries.connect(self.opSlicBoundariesCache.Output)

    def setupOutputs(self):        
        # The cache is capable of requesting and storing results in small blocks,
        # but we want to force the entire image to be handled and stored at once.
        # Therefore, we set the 'block shape' to be the entire image -- there will only be one block stored in the cache.
        # (Note: The OpBlockedArrayCache.innerBlockshape slot is deprecated and ignored.)
        self.opSlicCache.outerBlockShape.setValue( self.Input.meta.shape )
        self.opSlicBoundariesCache.outerBlockShape.setValue( self.Input.meta.shape )

        def markAllOutputsDirty( *args ):
            self.propagateDirty( self.Input, (), slice(None) )
        
    
    def execute(self, slot, subindex, roi, result):
        """
        Compute SLIC superpixel segmentation
        """      
        pass # shouldn't do anything, as the computation should be done in OpSlic
        
    def propagateDirty(self, slot, subindex, roi):
        pass # nothing to do here, likewise
     


