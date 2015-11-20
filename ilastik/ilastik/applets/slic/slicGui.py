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
        self._drawer.MaxIterSpinBox.valueChanged.connect(self.updateOperatorMaxIter)

        def updateDrawerFromOperator():
            SuperPixelSize, Cubeness, MaxIter = (10,10, 5)

            if self.topLevelOperatorView.SuperPixelSize.ready():
                SuperPixelSize = self.topLevelOperatorView.SuperPixelSize.value

            if self.topLevelOperatorView.Cubeness.ready():
                Cubeness = self.topLevelOperatorView.Cubeness.value

            if self.topLevelOperatorView.MaxIter.ready():
                MaxIter = self.topLevelOperatorView.MaxIter.value

            self._drawer.SuperPixelSizeSpinBox.setValue(SuperPixelSize)
            self._drawer.CubenessSpinBox.setValue(Cubeness)
            self._drawer.MaxIterSpinBox.setValue(MaxIter)

        # If the operator is changed *outside* the GUI (e.g. the project is loaded),
        #  then update the GUI to match the new operator slot values.            
        self.topLevelOperatorView.SuperPixelSize.notifyDirty( bind(updateDrawerFromOperator))
        self.topLevelOperatorView.Cubeness.notifyDirty( bind(updateDrawerFromOperator))
        self.topLevelOperatorView.MaxIter.notifyDirty( bind(updateDrawerFromOperator))

        # Initialize the GUI with the operator's initial state.
        updateDrawerFromOperator()
        
    def getAppletDrawerUi(self):
        return self._drawer

    def updateOperatorSuperPixelSize(self, SuperPixelSize):
        self.topLevelOperatorView.SuperPixelSize.setValue(SuperPixelSize)
    
    def updateOperatorCubeness(self, Cubeness):
        self.topLevelOperatorView.Cubeness.setValue(Cubeness)
    
    def updateOperatorMaxIter(self, MaxIter):
        self.topLevelOperatorView.MaxIter.setValue(MaxIter)
    
    def setupLayers(self):
        """
        The LayerViewer base class calls this function to obtain the list of layers that 
        should be displayed in the central viewer.
        """
        layers = []

        outputImageSlot = self.topLevelOperatorView.Output
        if outputImageSlot.ready(): 
            outputLayer = self.createStandardLayerFromSlot( outputImageSlot )
            outputLayer.name = "SegmentedImage"
            outputLayer.visible = True
            outputLayer.opacity = 0.4
            layers.append(outputLayer)   

        boundariesImageSlot = self.topLevelOperatorView.Boundaries
        if boundariesImageSlot.ready(): 
            boundaryLayer = self.createStandardLayerFromSlot( boundariesImageSlot )
            boundaryLayer.name = "Boundaries"
            boundaryLayer.visible = True
            boundaryLayer.opacity = 0.4
            layers.append(boundaryLayer)
 
        inputImageSlot = self.topLevelOperatorView.Input 
        if inputImageSlot.ready():
            inputLayer = self.createStandardLayerFromSlot( inputImageSlot )
            inputLayer.name = "Input"
            inputLayer.visible = True
            inputLayer.opacity = 1.0
            layers.append(inputLayer)

        return layers