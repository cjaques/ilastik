from ilastik.applets.base.standardApplet import StandardApplet

from Arteta.opCountingArteta import OpCountingArteta
from countingSerializer import CountingSerializer

class CountingArtetaApplet(StandardApplet):
    def __init__(self,
                 name="Counting",
                 workflow=None,
                 projectFileGroupName="Counting"):
        self._topLevelOperator = OpCountingArteta(parent=workflow)
        super(CountingArtetaApplet, self).__init__(name=name, workflow=workflow)

        self._serializableItems = [CountingSerializer(self._topLevelOperator, projectFileGroupName)]   # Legacy (v0.5) importer
        self.predictionSerializer = self._serializableItems[0]

    @property
    def topLevelOperator(self):
        return self._topLevelOperator

    @property
    def dataSerializers(self):
        return self._serializableItems

    @property
    def singleLaneGuiClass(self):
        from countingArtetaGui import CountingArtetaGui
        return CountingArtetaGui