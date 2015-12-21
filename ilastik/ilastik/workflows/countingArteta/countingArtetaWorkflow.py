import sys
import os
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
from ilastik.applets.counting import CountingArtetaApplet

from lazyflow.graph import Graph
from lazyflow.roi import TinyVector, fullSlicing

class CountingArtetaWorkflow(Workflow):
    
    workflowName = "Objects counting - Arteta"
    workflowDescription = "Arteta-Lempitsky object counting"
    defaultAppletIndex = 1 # show DataSelection by default

    def __init__(self, shell, headless, workflow_cmdline_args, project_creation_args, appendBatchOperators=True, *args, **kwargs):
        # Create a graph to be shared by all operators
        graph = kwargs['graph'] if 'graph' in kwargs else Graph()
        if 'graph' in kwargs: del kwargs['graph']
        super( CountingArtetaWorkflow, self ).__init__( shell, headless, workflow_cmdline_args, project_creation_args, graph=graph, *args, **kwargs )
        self.stored_classifer = None
        self._workflow_cmdline_args = workflow_cmdline_args
        
        # Applets for workflow 
        self.projectMetadataApplet = ProjectMetadataApplet()
        
        self.dataSelectionApplet = DataSelectionApplet(self,
                                                       "Input Data",
                                                       "Input Data" )
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        role_names = ['Raw Data']
        opDataSelection.DatasetRoles.setValue( role_names )

        self.featureSelectionApplet = FeatureSelectionApplet(self,
                                                             "Feature Selection",
                                                             "FeatureSelections")

        self.countingApplet = CountingArtetaApplet(workflow=self)
        opCounting = self.countingApplet.topLevelOperator

        self._applets = []
        self._applets.append(self.projectMetadataApplet)
        self._applets.append(self.dataSelectionApplet)
        self._applets.append(self.featureSelectionApplet)
        self._applets.append(self.countingApplet)

    def connectLane(self, laneIndex):
    	# operators views
        opDataSelection = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opFeatureSelection = self.featureSelectionApplet.topLevelOperator.getLane(laneIndex)
        opCounting = self.countingApplet.topLevelOperator.getLane(laneIndex)

        # connecting lanes Data --> Features, Features --> Counting
        opFeatureSelection.InputImage.connect(opDataSelection.Image)
        opCounting.LabelsAllowedFlags.connect(opDataSelection.AllowLabels)
        opCounting.InputImages.connect(opDataSelection.Image)
        opCounting.FeatureImages.connect(opFeatureSelection.OutputImage)
        opCounting.LabelsAllowedFlags.connect(opDataSelection.AllowLabels)
        opCounting.CachedFeatureImages.connect( opFeatureSelection.CachedOutputImage )

    def prepareForNewLane(self, laneIndex):
        """
        Overridden from Workflow base class.
        Called immediately before a new lane is added to the workflow.
        """
        # When the new lane is added, dirty notifications will propagate throughout the entire graph.
        # This means the classifier will be marked 'dirty' even though it is still usable.
        # Before that happens, let's store the classifier, so we can restore it at the end of connectLane(), below.
        opCounting = self.countingApplet.topLevelOperator
        if opCounting.classifier_cache.Output.ready() and \
           not opCounting.classifier_cache._dirty:
            self.stored_classifer = opCounting.classifier_cache.Output.value
        else:
            self.stored_classifer = None

    def handleAppletStateUpdateRequested(self): 
        """
        Overridden from Workflow base class
        Called when an applet has fired the :py:attr:`Applet.statusUpdateSignal`
        """
        # If no data, nothing else is ready.
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        input_ready = len(opDataSelection.ImageGroup) > 0

        opFeatureSelection = self.featureSelectionApplet.topLevelOperator
        featureOutput = opFeatureSelection.OutputImage
        features_ready = input_ready and \
                         len(featureOutput) > 0 and  \
                         featureOutput[0].ready() and \
                         (TinyVector(featureOutput[0].meta.shape) > 0).all()

        self._shell.setAppletEnabled(self.featureSelectionApplet, input_ready)
        self._shell.setAppletEnabled(self.countingApplet, features_ready)

        # Lastly, check for certain "busy" conditions, during which we 
        #  should prevent the shell from closing the project.
        busy = False
        busy |= self.dataSelectionApplet.busy
        busy |= self.featureSelectionApplet.busy
        self._shell.enableProjectChanges( not busy )

    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName

