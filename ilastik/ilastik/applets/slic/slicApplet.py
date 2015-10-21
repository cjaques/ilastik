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
import os
from ilastik.applets.base.standardApplet import StandardApplet
from opSlic import OpSlic
from opCachedSlic import OpCachedSlic
from slicSerializer import SlicSerializer

class SlicApplet( StandardApplet ):
    """
    This applet runs the SLIC algorithm to compute superpixels or supervoxels
    """
    def __init__( self, workflow,projectFileGroupName ):

        # Multi-image operator
        # Debug variable, will disappear
        use_cache =  (os.environ.get('USE_SLIC_CACHED',False) == "1")
        
        if(use_cache):
            self._topLevelOperator = OpCachedSlic(parent=workflow)
        else:
            self._topLevelOperator = OpSlic(parent=workflow) 
        
        # Base class
        super(SlicApplet, self).__init__("Slic", workflow)
        self._serializableItems = []#SlicSerializer(self._topLevelOperator,projectFileGroupName)]
    
    def resetComputed(self):
        self._topLevelOperator.resetComputed()

    @property
    def topLevelOperator(self):
        return self._topLevelOperator
    
    @property
    def singleLaneGuiClass(self):
        from slicGui import SlicGui
        return SlicGui

    @property
    def broadcastingSlots(self):
        return []
    
    @property
    def dataSerializers(self):
        return self._serializableItems
