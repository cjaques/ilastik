from lazyflow.graph import Operator, InputSlot, OutputSlot
from functools import partial
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.request import Request
from lazyflow.operators import OpArrayCache

from ilastik.applets.base.applet import DatasetConstraintError
from ilastik.utility import MultiLaneOperatorABC, OperatorSubView

import traceback
from opSlic import OpSlic 
import numpy, vigra

class OpCachedSlic(Operator):
    """
    This is the cached operator for the SLIC class. It shouldn't compute anything, 
    just redirect computation to OpSlic and store the results in a cache 
    """
    name = "OpCachedSlic"
    category = "top-level"

    InputImages = InputSlot(level=1) # level = 1 so that input can be multiple images 
    SuperPixelSize = InputSlot(optional=True)
    Cubeness = InputSlot(optional=True)

    OutputImages = OutputSlot(level=1) # Result of the SLIC algorithm goes in this images

    def ResetComputed(self):
        pass

    def setupOutputs(self):        
        # Copy the meta info from each input to the corresponding output
        self.OutputImages.resize( len(self.InputImages) )
        for index, islot in enumerate(self.InputImages):
            self.OutputImages[index].meta.assignFrom(islot.meta)
        self.OutputImages.meta.assignFrom(self.InputImages.meta)

        def markAllOutputsDirty( *args ):
            self.propagateDirty( self.InputImages, (), slice(None) )
        self.InputImages.notifyInserted( markAllOutputsDirty )
        self.InputImages.notifyRemoved( markAllOutputsDirty )

    def __init__(self, *args, **kwargs):
        super( OpCachedSlic, self ).__init__(*args, **kwargs)
        # OpSlic to compute SLIC 
        self._opSlic = OpSlic(parent=self )
        self._opSlic.InputVolume.connect(self.InputImages) 
        # The line above blocks the ready signal of the outputslot in SlicGui.py, why?
        # this makes that nothing appears in the layers of the viewer...
        self._opSlic.Cubeness.connect(self.Cubeness)
        self._opSlic.SuperPixelSize.connect(self.SuperPixelSize)
        self.OutputImages.connect(self._opSlic.SegmentedImage)
        # OpSlicCache to cache results (operator Wrapper to promote OpSlicCache.input to level=1) 
        self._opSlicCache = OperatorWrapper(OpArrayCache,parent=self,promotedSlotNames=['Input']) 
        # Cleanup of wrapped cache doesn't work 
        self._opSlicCache.Input.connect(self._opSlic.SegmentedImage)

        self.setupOutputs()

    def execute(self, slot, subindex, roi, result):
        """
        Compute SLIC superpixel segmentation
        """      
        pass # shouldn't do anything, as the computation should be done in OpSlic
        
    def propagateDirty(self, slot, subindex, roi):
        pass # nothing to do here, likewise
        
    #############################################
    ## Methods to satisfy MultiLaneOperatorABC ##
    #############################################

    def addLane(self, laneIndex):
        """
        Add an image lane to the top-level operator.
        """
        numLanes = len(self.InputImages)
        assert numLanes == laneIndex, "Image lanes must be appended."        
        self.InputImages.resize(numLanes+1) 
        self._opSlic.addLane(laneIndex) # we have to pass this to _opSlic, and it breaks the assert numlane == laneIndex
                                        # this is why it is commented in the addlane function of OpSlic
        
    def removeLane(self, laneIndex, finalLength):
        """
        Remove the specified image lane from the top-level operator.
        """
        self.InputImages.removeSlot(laneIndex, finalLength)
        self._opSlic.removeLane(laneIndex,finalLength) # we have to pass this to _opSlic

    def getLane(self, laneIndex):
        return OperatorSubView(self, laneIndex)

assert issubclass(OpCachedSlic, MultiLaneOperatorABC)

