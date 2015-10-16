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
from opMitochondriaSegmentationDataExport import OpMitochondriaSegmentationDataExport
from ilastik.applets.dataExport.dataExportApplet import DataExportApplet
from ilastik.applets.dataExport.dataExportSerializer import DataExportSerializer

from ilastik.utility import OpMultiLaneWrapper

class MitochondriaSegmentationDataExportApplet( DataExportApplet ):
    """
    This a specialization of the generic data export applet that
    provides a special viewer for mitochondria segmentation.
    """
    def __init__( self, workflow, title, isBatch=False ):
        # Our operator is a subclass of the generic data export operator
        self._topLevelOperator = OpMultiLaneWrapper( OpMitochondriaSegmentationDataExport, parent=workflow,
                                     promotedSlotNames=set(['RawData', 'Inputs', 'RawDatasetInfo', 'ConstraintDataset']) )
        self._gui = None
        self._title = title
        self._serializers = [ DataExportSerializer(self._topLevelOperator, title) ]

        # Base class init
        super(MitochondriaSegmentationDataExportApplet, self).__init__(workflow, title, isBatch)
        
    @property
    def dataSerializers(self):
        return self._serializers

    @property
    def topLevelOperator(self):
        return self._topLevelOperator

    def getMultiLaneGui(self):
        if self._gui is None:
            # Gui is a special subclass of the generic gui
            from mitochondriaSegmentationDataExportGui import MitochondriaSegmentationDataExportGui
            self._gui = MitochondriaSegmentationDataExportGui( self, self.topLevelOperator )
        return self._gui





