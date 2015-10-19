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
from lazyflow.request import Request

from ilastik.applets.base.applet import DatasetConstraintError
from ilastik.utility import MultiLaneOperatorABC, OperatorSubView


import sys
sys.path.append('/Users/Chris/Code/python_tests/slic/')
import slic # as slic
import numpy, vigra

class OpSlic(Operator):
    """
    This is the default top-level operator for the SLIC class.
    """
    name = "OpSlic"
    category = "top-level"

    InputVolume = InputSlot(level=1) # level = 1 so that input can be multiple images 
    SuperPixelSize = InputSlot(optional=True)
    Cubeness = InputSlot(optional=True)
    
    #OtherInput = InputSlot(optional=True)

    SegmentedImage = OutputSlot(level=1) # How to set its shape , only 1 image with "pixel" value being equal to cluster center number? level = 1?

    def __init__(self, *args, **kwargs):
        super( OpSlic, self ).__init__(*args, **kwargs)

    def setupOutputs(self):
        # check input shape and assign output
        # shape = self.InputVolume.meta.shape
        
        # Copy the meta info from each input to the corresponding output --> this has to be done to display data    
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
            
            region = self.InputVolume[0].get(roi).wait()
            result = numpy.zeros( region.shape,dtype=numpy.float32)

            result = slic.ArgsTest(region,region.shape[0],region.shape[1])
            # print 'SLIC code executed ---=---'
            # print 'Newval ------'
            # print newVal.shape
            # print newVal[(10,10,0)]
            # print 'Res  ------  '
            # print region.shape
            # print region[(10,10,0)]
            # # result = numpy.array(region.shape)
            # r2 = Request ( partial(slic.ComputeSlicXD,region))
            # result = r1.wait()
        else: 
            result=self.InputVolume
        # slic.ComputeSlicXD(region) #, result) 


        return result

    def propagateDirty(self, slot, subindex, roi):
        # If the dirty slot is one of our two constants, then the entire image region is dirty
        if slot == self.InputVolume:
            roi = slice(None) # The whole image region
        
        # All inputs affect all outputs, so every image is dirty now
        for oslot in self.SegmentedImage:
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

assert issubclass(OpSlic, MultiLaneOperatorABC)


    # def checkConstraints(self, *args):
    #     """
    #     Example of how to check input data constraints.
    #     """
    #     if self.OtherInput.ready() and self.RawInput.ready():
    #         rawTaggedShape = self.RawInput.meta.getTaggedShape()
    #         otherTaggedShape = self.OtherInput.meta.getTaggedShape()
    #         raw_time_size = rawTaggedShape.get('t', 1)
    #         other_time_size = otherTaggedShape.get('t', 1)
    #         if raw_time_size != other_time_size and raw_time_size != 1 and other_time_size != 1:
    #             msg = "Your 'raw' and 'other' datasets appear to have differing sizes in the time dimension.\n"\
    #                   "Your datasets have shapes: {} and {}".format( self.RawInput.meta.shape, self.OtherInput.meta.shape )
    #             raise DatasetConstraintError( "Layer Viewer", msg )
                
    #         rawTaggedShape['c'] = None
    #         otherTaggedShape['c'] = None
    #         rawTaggedShape['t'] = None
    #         otherTaggedShape['t'] = None
    #         if dict(rawTaggedShape) != dict(otherTaggedShape):
    #             msg = "Raw data and other data must have equal spatial dimensions (different channels are okay).\n"\
    #                   "Your datasets have shapes: {} and {}".format( self.RawInput.meta.shape, self.OtherInput.meta.shape )
    #             raise DatasetConstraintError( "Layer Viewer", msg )
        
        