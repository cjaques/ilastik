from ilastik.applets.base.appletSerializer import AppletSerializer, SerialSlot

class DummySerializer(AppletSerializer):
    """
    Serializes the user's settings in the "Dummy" applet to an ilastik v0.6 project file.
    """
    def __init__(self, operator, projectFileGroupName):
        slots = [SerialSlot(operator.ScalingFactor, selfdepends=True),
                 SerialSlot(operator.Offset, selfdepends=True),
                 SerialSlot(operator.GaussianKernelSize,selfdepends=True)]
        
        super(DummySerializer, self).__init__(projectFileGroupName,
                                                         slots=slots)
