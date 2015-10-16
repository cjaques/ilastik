# Useless plugin developped as a training

from ilastik.workflow import Workflow
from lazyflow.graph import Graph
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.dummy.dummyApplet import DummyApplet
from ilastik.applets.dataExport.dataExportApplet import DataExportApplet

class dummyWorkflow(Workflow):
    def __init__(self, shell, headless, workflow_cmdline_args, project_creation_args):
        # Create a graph to be shared by all operators
        graph = Graph()
        super(dummyWorkflow, self).__init__(shell, headless, workflow_cmdline_args, project_creation_args, graph=graph)
        self._applets = []

        # Create applets 
        self.dataSelectionApplet = DataSelectionApplet(self, "Input Data", "Input Data", supportIlastik05Import=True) #, batchDataGui=False)
        self.dummyApplet = DummyApplet(self, "Dummy") # Custom applet, 3D gaussian 
        self.dataExportApplet = DataExportApplet(self, "Data Export")
        
        opDataExport = self.dataExportApplet.topLevelOperator
        opDataExport.SelectionNames.setValue( ['Raw Data', 'Other Data'] )

        opDataSelection = self.dataSelectionApplet.topLevelOperator
        opDataSelection.DatasetRoles.setValue( ['Raw Data', 'Other Data'] )

        self._applets.append( self.dataSelectionApplet )
        self._applets.append( self.dummyApplet )
        self._applets.append( self.dataExportApplet )

    def connectLane(self, laneIndex):
        opDataSelectionView = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opDummy = self.dummyApplet.topLevelOperator.getLane(laneIndex)
        opDataExportView = self.dataExportApplet.topLevelOperator.getLane(laneIndex)

        # Connect top-level operators
        opDummy.Input.connect( opDataSelectionView.ImageGroup[0])

        # Connect exportView to lanes
        opDataExportView.RawData.connect( opDataSelectionView.ImageGroup[0] )
        opDataExportView.RawDatasetInfo.connect( opDataSelectionView.DatasetGroup[0] )        
        opDataExportView.WorkingDirectory.connect( opDataSelectionView.WorkingDirectory )

        opDataExportView.Inputs.resize(2)
        opDataExportView.Inputs[0].connect( opDataSelectionView.ImageGroup[0] )
        opDataExportView.Inputs[1].connect( opDataSelectionView.ImageGroup[1] )

    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName
