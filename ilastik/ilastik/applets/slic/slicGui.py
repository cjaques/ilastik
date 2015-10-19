from PyQt4 import uic

import os
from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from ilastik.utility.gui import ThreadRouter, threadRouted

from ilastik.utility import bind

class SlicGui(LayerViewerGui):
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
        super(SlicGui, self).__init__(parentApplet, topLevelOperatorView)
            
    def initAppletDrawerUi(self):
        # Load the ui file (find it in our own directory)
        localDir = os.path.split(__file__)[0]
        self._drawer = uic.loadUi(localDir+"/drawer.ui")

        # If the user changes a setting the GUI, update the appropriate operator slot.        
        self._drawer.SuperPixelSizeSpinBox.valueChanged.connect(self.updateOperatorSuperPixelSize)
        self._drawer.CubenessSpinBox.valueChanged.connect(self.updateOperatorCubeness)

        def updateDrawerFromOperator():
            SuperPixelSize, Cubeness = (20,5)

            if self.topLevelOperatorView.SuperPixelSize.ready():
                SuperPixelSize = self.topLevelOperatorView.SuperPixelSize.value

            if self.topLevelOperatorView.Cubeness.ready():
                Cubeness = self.topLevelOperatorView.Cubeness.value

            self._drawer.SuperPixelSizeSpinBox.setValue(SuperPixelSize)
            self._drawer.CubenessSpinBox.setValue(Cubeness)

        # If the operator is changed *outside* the GUI (e.g. the project is loaded),
        #  then update the GUI to match the new operator slot values.            
        self.topLevelOperatorView.SuperPixelSize.notifyDirty( bind(updateDrawerFromOperator) )
        self.topLevelOperatorView.Cubeness.notifyDirty( bind(updateDrawerFromOperator))

        # Initialize the GUI with the operator's initial state.
        updateDrawerFromOperator()

        # Provide defaults if the operator isn't already configured.
        #  (e.g. if it's a blank project, then the operator won't have any setup yet.)
        if not self.topLevelOperatorView.SuperPixelSize.ready():
            self.updateOperatorSuperPixelSize(10)
        if not self.topLevelOperatorView.Cubeness.ready():
            self.updateOperatorCubeness(5)
        
    def updateOperatorSuperPixelSize(self, SuperPixelSize):
        self.topLevelOperatorView.SuperPixelSize.setValue(SuperPixelSize)
    
    # @threadRouted
    # def updateAllLayers(self, slot=None): #overload to avoid tiles.
    #     self._need_update = False
    
    def getAppletDrawerUi(self):
        return self._drawer

    def updateOperatorCubeness(self, Cubeness):
        self.topLevelOperatorView.Cubeness.setValue(Cubeness)
    
    def setupLayers(self):
        """
        The LayerViewer base class calls this function to obtain the list of layers that 
        should be displayed in the central viewer.
        """
        layers = []

        # Show the Output data
        outputImageSlot = self.topLevelOperatorView.SegmentedImage
        if outputImageSlot.ready():
            outputLayer = self.createStandardLayerFromSlot( outputImageSlot )
            outputLayer.name = "SegmentedImage"
            outputLayer.visible = True
            outputLayer.opacity = 0.8
            layers.append(outputLayer)
            
        # Show the raw input data as a convenience for the user
        inputImageSlot = self.topLevelOperatorView.InputVolume
        if inputImageSlot.ready():
            inputLayer = self.createStandardLayerFromSlot( inputImageSlot )
            inputLayer.name = "Input"
            inputLayer.visible = True
            inputLayer.opacity = 1.0
            layers.append(inputLayer)

        return layers