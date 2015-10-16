from ilastik.applets.base.standardApplet import StandardApplet
from opDummy import OpDummy
from dummySerializer import DummySerializer 

class DummyApplet( StandardApplet ):
    """
    This applet is a test generates 3D weighted gaussian of input 
    The GUI is not aware of multiple image lanes (it is written as if the applet were single-image only).
    The top-level operator is explicitly multi-image (it is not wrapped in an operatorwrapper).
    """
    def __init__( self, workflow, projectFileGroupName ):
        # Multi-image operator
        self._topLevelOperator = OpDummy(parent=workflow)
        
        # Base class
        super(DummyApplet, self).__init__( "Dummy ", workflow )
        self._serializableItems = [ DummySerializer( self._topLevelOperator, projectFileGroupName ) ]
            
    @property
    def topLevelOperator(self):
        return self._topLevelOperator

    @property
    def singleLaneGuiClass(self):
        from dummyGui import DummyGui
        return DummyGui

    @property
    def dataSerializers(self):
        return self._serializableItems