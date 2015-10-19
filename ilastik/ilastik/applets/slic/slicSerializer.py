from ilastik.applets.base.appletSerializer import AppletSerializer, SerialSlot

class SlicSerializer(AppletSerializer):
    """
    Serializes the user's settings in the "SLIC" applet to an ilastik v0.6 project file.
    """
    def __init__(self, operator, projectFileGroupName):
        slots = [SerialSlot(operator.SuperPixelSize, selfdepends=True),
                 SerialSlot(operator.Cubeness, selfdepends=True)]
        
        super(SlicSerializer, self).__init__(projectFileGroupName,
                                                         slots=slots)
