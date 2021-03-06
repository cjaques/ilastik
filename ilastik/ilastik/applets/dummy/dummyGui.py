from PyQt4 import uic

import os
from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui

from ilastik.utility import bind

class DummyGui(LayerViewerGui):
    """
    """
    
    ###########################################
    ### AppletGuiInterface Concrete Methods ###
    ###########################################
    
    def appletDrawer(self):
        return self.getAppletDrawerUi()

    # (Other methods already provided by our base class)

    ###########################################
    ###########################################
    
    def __init__(self, parentApplet, topLevelOperatorView):
        """
        """
        self.topLevelOperatorView = topLevelOperatorView
        super(DummyGui, self).__init__(parentApplet, topLevelOperatorView)
            
    def initAppletDrawerUi(self):
        # Load the ui file (find it in our own directory)
        localDir = os.path.split(__file__)[0]
        self._drawer = uic.loadUi(localDir+"/drawer.ui")

        # If the user changes a setting the GUI, update the appropriate operator slot.        
        self._drawer.scalingFactorSpinBox.valueChanged.connect(self.updateOperatorScalingFactor)
        self._drawer.offsetSpinBox.valueChanged.connect(self.updateOperatorOffset)
        self._drawer.gaussianKernelSizeSpinBox.valueChanged.connect(self.updateOperatorGaussianKernelSize)

        def updateDrawerFromOperator():
            scalingFactor, offset, gaussianKernelSize = (0,0,5)

            if self.topLevelOperatorView.ScalingFactor.ready():
                scalingFactor = self.topLevelOperatorView.ScalingFactor.value
            if self.topLevelOperatorView.Offset.ready():
                offset = self.topLevelOperatorView.Offset.value
            if self.topLevelOperatorView.GaussianKernelSize.ready():
                gaussianKernelSize = self.topLevelOperatorView.GaussianKernelSize.value

            self._drawer.scalingFactorSpinBox.setValue(scalingFactor)
            self._drawer.offsetSpinBox.setValue(offset)
            self._drawer.gaussianKernelSizeSpinBox.setValue(gaussianKernelSize)

        # If the operator is changed *outside* the GUI (e.g. the project is loaded),
        #  then update the GUI to match the new operator slot values.            
        self.topLevelOperatorView.ScalingFactor.notifyDirty( bind(updateDrawerFromOperator) )
        self.topLevelOperatorView.Offset.notifyDirty( bind(updateDrawerFromOperator) )
        self.topLevelOperatorView.GaussianKernelSize.notifyDirty( bind(updateDrawerFromOperator))

        # Initialize the GUI with the operator's initial state.
        updateDrawerFromOperator()

        # Provide defaults if the operator isn't already configured.
        #  (e.g. if it's a blank project, then the operator won't have any setup yet.)
        if not self.topLevelOperatorView.ScalingFactor.ready():
            self.updateOperatorScalingFactor(1)
        if not self.topLevelOperatorView.Offset.ready():
            self.updateOperatorOffset(0)
        if not self.topLevelOperatorView.GaussianKernelSize.ready():
            self.updateOperatorGaussianKernelSize(5)
        
    def updateOperatorScalingFactor(self, scalingFactor):
        self.topLevelOperatorView.ScalingFactor.setValue(scalingFactor)
    
    def updateOperatorOffset(self, offset):
        self.topLevelOperatorView.Offset.setValue(offset)
    
    def getAppletDrawerUi(self):
        return self._drawer

    def updateOperatorGaussianKernelSize(self, gaussianKernelSize):
        self.topLevelOperatorView.GaussianKernelSize.setValue(gaussianKernelSize)
    
    def setupLayers(self):
        """
        The LayerViewer base class calls this function to obtain the list of layers that 
        should be displayed in the central viewer.
        """
        layers = []

        # Show the Output data
        outputImageSlot = self.topLevelOperatorView.Output
        if outputImageSlot.ready():
            outputLayer = self.createStandardLayerFromSlot( outputImageSlot )
            outputLayer.name = "Dummy_output"
            outputLayer.visible = True
            outputLayer.opacity = 1.0
            layers.append(outputLayer)
            
        # Show the raw input data as a convenience for the user
        inputImageSlot = self.topLevelOperatorView.Input
        if inputImageSlot.ready():
            inputLayer = self.createStandardLayerFromSlot( inputImageSlot )
            inputLayer.name = "Input"
            inputLayer.visible = True
            inputLayer.opacity = 1.0
            layers.append(inputLayer)

        return layers