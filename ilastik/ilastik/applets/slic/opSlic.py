###############################################################################
#   ilastik: interactive learning and segmentation toolkit
#
#       Copyright (C) 2011-2014, the ilastik developers
#                                <team@ilastik.org>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# In addition, as a special exception, the copyright holders of
# ilastik give you permission to combine ilastik with applets,
# workflows and plugins which are not covered under the GNU
# General Public License.
#
# See the LICENSE file for details. License information is also available
# on the ilastik web site at:
#		   http://ilastik.org/license.html
###############################################################################
from lazyflow.graph import Operator, InputSlot, OutputSlot
from functools import partial
from lazyflow import operatorWrapper, roi as ROI
from lazyflow.request import Request
from lazyflow.operators import OpArrayCache
from lazyflow.operatorWrapper import OperatorWrapper
from lazyflow.rtype import SubRegion

from ilastik.applets.base.applet import DatasetConstraintError
from ilastik.utility import MultiLaneOperatorABC, OperatorSubView

import traceback
import os
import sys
sys.path.append('/Users/Chris/Code/python_tests/slic/') #TODO CHRis - automate this (integrate slic code in Ilastik?)
import slic 
import numpy, vigra

class OpSlic(Operator):
    """
    This is the default operator for the SLIC implementation.
    """
    name = "OpSlic"
    category = "top-level"

    InputVolume = InputSlot(level=1) # level = 1 so that input can be multiple images 
    SuperPixelSize = InputSlot(optional=True)
    Cubeness = InputSlot(optional=True)

    SegmentedImage = OutputSlot(level=1) 
    
    def __init__(self, *args, **kwargs):
        super( OpSlic, self ).__init__(*args, **kwargs)
        self._computed = False 

    def ResetComputed(self):
        self._computed = False

    def setupOutputs(self):   
        # Copy the meta info from each input to the corresponding output
        self.SegmentedImage.resize( len(self.InputVolume) )
        for index, islot in enumerate(self.InputVolume):
            self.SegmentedImage[index].meta.assignFrom(islot.meta)

        self.SegmentedImage.meta.assignFrom(self.InputVolume.meta)

        def markAllOutputsDirty( *args ):
            self.propagateDirty( self.InputVolume, (), slice(None) )
        self.InputVolume.notifyInserted( markAllOutputsDirty )
        self.InputVolume.notifyRemoved( markAllOutputsDirty )

    def execute(self, slot, subindex, roi, result):
        """
        Compute SLIC superpixel segmentation
        """        
        if slot==self.SegmentedImage:
            if(self._computed == False): # self._computed = true means we already computed SLIC on the whole volume
                shape = self.InputVolume[0].meta.shape
                newRoi = SubRegion(self.InputVolume[0], start=tuple( (0,0,0) ),stop=tuple( (shape[0],shape[1],shape[2]) )) # newRoi is the whole input
                region = self.InputVolume[0].get(newRoi).wait()
                self.resultCache = slic.ArgsTest(region,region.shape[0],region.shape[1], self.SuperPixelSize.value ,self.Cubeness.value)
                self._computed = True

            result = self.resultCache[roi.start[0]:roi.stop[0],roi.start[1]:roi.stop[1],roi.start[2]:roi.stop[2]]
        else: 
            result=self.InputVolume

        return result

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
        # assert numLanes == laneIndex, "Image lanes must be appended."  # this happen not to be the case with cache... why?
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

assert issubclass(OpSlic, MultiLaneOperatorABC)

        
        