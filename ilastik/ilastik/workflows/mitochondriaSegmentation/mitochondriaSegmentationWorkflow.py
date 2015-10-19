###############################################################################
# Mitochondria Segmentation Workflow
# from 
###############################################################################
import sys
import copy
import argparse
import logging
logger = logging.getLogger(__name__)

import numpy

from ilastik.config import cfg as ilastik_config
from ilastik.workflow import Workflow
from ilastik.applets.projectMetadata import ProjectMetadataApplet
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.featureSelection import FeatureSelectionApplet
# from ilastik.applets.mitochondriaSegmentation import MitochondriaSegmentationApplet, MitochondriaSegmentationDataExportApplet
# from ilastik.applets.batchProcessing import BatchProcessingApplet

from ilastik.applets.slic import SlicApplet

from lazyflow.graph import Graph
from lazyflow.roi import TinyVector, fullSlicing

class MitochondriaSegmentationWorkflow(Workflow):
    
    workflowName = "Mitochondria Segmentation"
    workflowDescription = "Segments mitochondria using contextual cues"
    defaultAppletIndex = 1 # show DataSelection by default
    
    DATA_ROLE_RAW = 0
    DATA_ROLE_PREDICTION_MASK = 1
    
    EXPORT_NAMES = ['Probabilities', 'Simple Segmentation', 'Uncertainty', 'Features'] # TODO : is Features really necessary or enough?
    
    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName

    def __init__(self, shell, headless, workflow_cmdline_args, project_creation_args, appendBatchOperators=True, *args, **kwargs):
        # Create a graph to be shared by all operators
        graph = Graph()
        super( MitochondriaSegmentationWorkflow, self ).__init__( shell, headless, workflow_cmdline_args, project_creation_args, graph=graph, *args, **kwargs )
        self.stored_classifer = None
        self._applets = []
        self._workflow_cmdline_args = workflow_cmdline_args

        
        # Applets for training (interactive) workflow 
        self.projectMetadataApplet = ProjectMetadataApplet()
        
        self.dataSelectionApplet = self.createDataSelectionApplet()
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        
        # see role constants, above
        role_names = ['Raw Data', 'Prediction Mask']
        opDataSelection.DatasetRoles.setValue( role_names )

        # self.msApplet = self.createMitochondriaSegmentationApplet() # TODO : factory to create applet
        # self.msApplet = self.createMitochondriaSegmentationApplet()	# TODO CHRIS : create applet + replace old one (*msApplet)
        self.slicApplet = self.createSlicSegmentationApplet()
        #opClassify = self.msApplet.topLevelOperator

        # self.dataExportApplet = MitochondriaSegmentationDataExportApplet(self, "Prediction Export")

        # Expose for shell
        self._applets.append(self.projectMetadataApplet)
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.slicApplet)
    #     logger.warn("Unused command-line args: {}".format( unused_args ))

    def createDataSelectionApplet(self):
        """
        Can be overridden by subclasses, if they want to use 
        special parameters to initialize the DataSelectionApplet.
        """
        data_instructions = "Select your input data using the 'Raw Data' tab shown on the right"
        return DataSelectionApplet( self,
                                    "Input Data",
                                    "Input Data",
                                    supportIlastik05Import=True,
                                    instructionText=data_instructions )


    

    def createSlicSegmentationApplet(self):
        """
        SLIC applet, based on RK's C++ implementation
        """
        print 'Creating SLIC segmentation applet'
        return SlicApplet( self) # "SlicSegmentation" )

 
    def connectLane(self, laneIndex):
        print '[MitochondriaSegmentation] connectLane '
        opDataSelectionView = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opSlicView = self.slicApplet.topLevelOperator.getLane(laneIndex)
        
        # Connect top-level operators                                                                                                                 
        opSlicView.InputVolume.connect( opDataSelectionView.Image )
        # opSlicView.OtherInput.connect( opDataSelectionView.ImageGroup[1] )

    def handleAppletStateUpdateRequested(self): 
        """
        Overridden from Workflow base class
        Called when an applet has fired the :py:attr:`Applet.statusUpdateSignal`
        """
        # If no data, nothing else is ready.
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        input_ready = len(opDataSelection.ImageGroup) > 0

        self._shell.setAppletEnabled(self.slicApplet, input_ready)
        
        # Lastly, check for certain "busy" conditions, during which we 
        #  should prevent the shell from closing the project.
        busy = False
        busy |= self.dataSelectionApplet.busy
        self._shell.enableProjectChanges( not busy )
    
    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName


  