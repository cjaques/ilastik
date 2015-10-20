from lazyflow.graph import Operator, InputSlot, OutputSlot
from functools import partial
from lazyflow import operatorWrapper
from lazyflow.request import Request
from lazyflow.operators import OpArrayCache

from ilastik.applets.base.applet import DatasetConstraintError
from ilastik.utility import MultiLaneOperatorABC, OperatorSubView

import traceback

import sys
sys.path.append('/Users/Chris/Code/python_tests/slic/') #TODO CHRis - automate this (integrate slic code in Ilastik?)
import slic 
import numpy, vigra

class OpCachedSlic(Operator):
    """
    This is the cached operator for the SLIC class.
    """
    name = "OpCachedSlic"
    category = "top-level"

    InputVolume = InputSlot(level=1) # level = 1 so that input can be multiple images 
    SuperPixelSize = InputSlot(optional=True)
    Cubeness = InputSlot(optional=True)

    SegmentedImage = OutputSlot(level=1) # How to set its shape , only 1 image with "pixel" value being equal to cluster center number? level = 1?

    def __init__(self, *args, **kwargs):
        super( OpCachedSlic, self ).__init__(*args, **kwargs)

        # OpSlic to compute SLIC
        self._opSlic = OpSlic(parent=self)
        self._opSlic.InputVolume.connect(self.InputVolume)
        self._opSlic.Cubeness.connect(self.Cubeness)
        self._opSlic.SuperPixelSize.connect(self.SuperPixelSize)

        # OpSlicCache to cache results
        self._opSlicCache = OpArrayCache(parent=self)

        # if( len(self._opSlic.SegmentedImage) > 0 ):
        #     print 'Ok ... '
        #     self._opSlicCache.Input.connect(self._opSlic.SegmentedImage)

    def setupOutputs(self):        
        # Copy the meta info from each input to the corresponding output
        self.SegmentedImage.meta.assignFrom(self._opSlicCache.Output.meta)
        self.SegmentedImage.resize( len(self.InputVolume) )
        for index, islot in enumerate(self.InputVolume):
            self.SegmentedImage[index].meta.assignFrom(islot.meta)

        def markAllOutputsDirty( *args ):
            self.propagateDirty( self.InputVolume, (), slice(None) )
        self.InputVolume.notifyInserted( markAllOutputsDirty )
        self.InputVolume.notifyRemoved( markAllOutputsDirty )


    def execute(self, slot, subindex, roi, result):
        """
        Compute SLIC superpixel segmentation
        """        
        if slot is self.SegmentedImage:
            print 'Slot is SegmentedImage'
            return self._opSlicCache.Output.get(roi).wait()
        else:
            print 'Slot is something else : ',slot
            return self.InputVolume.get(roi).wait()

    def propagateDirty(self, slot, subindex, roi):
        # If the dirty slot is one of our two constants, then the entire image region is dirty
        if slot == self.InputVolume:
            roi = slice(None) # The whole image region
        
        # All inputs affect all outputs, so every image is dirty now
        for oslot in self.SegmentedImage:
            roi = slice(None) # the whole image
            oslot.setDirty( roi )

    #############################################
    ## Methods to satisfy MultiLaneOperatorABC ##
    #############################################

    def addLane(self, laneIndex):
        """
        Add an image lane to the top-level operator.
        """
        numLanes = len(self.InputVolume)
        assert numLanes == laneIndex, "Image lanes must be appended."        
        self.InputVolume.resize(numLanes+1)
        self.SegmentedImage.resize(numLanes+1)
        
    def removeLane(self, laneIndex, finalLength):
        """
        Remove the specified image lane from the top-level operator.
        """
        self.InputVolume.removeSlot(laneIndex, finalLength)
        self.SegmentedImage.removeSlot(laneIndex, finalLength)

    def getLane(self, laneIndex):
        return OperatorSubView(self, laneIndex)

assert issubclass(OpCachedSlic, MultiLaneOperatorABC)