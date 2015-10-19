from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.rtype import SubRegion
import numpy as np 
from ilastik.utility import MultiLaneOperatorABC, OperatorSubView

# for debugging
import sys
import sys
sys.path.append('/Users/Chris/Code/python_tests/slic/')
import slic # as slic
import numpy, vigra


class OpDummy(Operator):
    """
    Multi-image operator.
    Takes 3D weighted gaussian average 
    Note: Inputs must all have the same shape.
    """    
    ScalingFactor = InputSlot() # Scale after subtraction
    Offset = InputSlot()        # Offset final results
    GaussianKernelSize = InputSlot()   # Size of the Gaussian kernel
    Input = InputSlot(level=1)  # Multi-image input (because of level=1, level=0 would have been single image input)

    Mean = OutputSlot()
    Output = OutputSlot(level=1) # Multi-image output

    alreadyCachedInput = False # to store computed results
    CachedInput = [0.0]
    ComputedROIs = []
    CachedResults = []#np.zeros(1) --> causes issues, too slow to resize
    InputShape = []

    def setupOutputs(self):
        # Ensure all inputs have the same shape
        if len(self.Input) > 0:
            shape = self.Input[0].meta.shape
            for islot in self.Input:
                if islot.meta.shape != shape:
                    raise RuntimeError("Input images must have the same shape.")

        # Copy the meta info from each input to the corresponding output        
        self.Output.resize( len(self.Input) )
        for index, islot in enumerate(self.Input):
            self.Output[index].meta.assignFrom(islot.meta)
            

        self.Mean.meta.assignFrom(self.Input[0].meta)

        def markAllOutputsDirty( *args ):
            self.propagateDirty( self.Input, (), slice(None) )
        self.Input.notifyInserted( markAllOutputsDirty )
        self.Input.notifyRemoved( markAllOutputsDirty )

    def execute(self, slot, subindex, roi, result):
        """
        Compute.  dummy operator execution
        """
        # ======================= INFOS =================================
        # total size of the input image : self.Input[0].meta.shape

        if(slot == self.Mean):
            region = self.Input[0].get(roi).wait()
            result = numpy.zeros( region.shape,dtype=numpy.float32)

            result = slic.ArgsTest(region,region.shape[0],region.shape[1])
            # print 'SLIC code executed ---=---'
            # print 'Newval ------'
            # print newVal.shape
            # print newVal[(10,10,0)]
            # print 'Res  ------  '
            # print region.shape
            # print region[(10,10,0)]
            # # result = numpy.array(region.shape)
            # r2 = Request ( partial(slic.ComputeSlicXD,region))
            # result = r1.wait()
        else: 
            result=self.Input
        # slic.ComputeSlicXD(region) #, result) 


        return result

        # # Cache computation
        # if (self.alreadyCachedInput == False):
        #     self.InputShape = self.Input[0].meta.shape 
        #     start = (0, 0, 0,0)
        #     stop = (self.InputShape[0], self.InputShape[1], self.InputShape[2],self.InputShape[3]) # pourquoi c = 3 ? (axis = zyxc)
        #     totalROI = SubRegion(self.Output,start=tuple(start), stop=tuple(stop)) 
        #     self.CachedInput[:] =  self.Input[0].get(totalROI).wait() # cache input, probablement pas necessaire ----> meilleur moyen d'acceder au images?
        #     self.CachedResults = np.resize(self.CachedResults,self.InputShape)
        #     self.CachedResults[:] = np.zeros(self.InputShape)
        #     self.alreadyCachedInput = True
        
        # results = np.zeros(self.InputShape)
        # # Results already computed ? --> will depend on roi ---> store ROIs in memory 
        # if roi in self.ComputedROIs:
        #     results[roi.start[0]:roi.stop[0], roi.start[1]:roi.stop[1], roi.start[2]:roi.stop[2],roi.start[3]:roi.stop[3]] = self.getCachedResults(roi)   #better way to index arrays. 
        # else:
        #     results[roi.start[0]:roi.stop[0], roi.start[1]:roi.stop[1], roi.start[2]:roi.stop[2],roi.start[3]:roi.stop[3]] = self.computeResults(roi) 
        #     if results is not None : # Fixme, this condition will always be true. 
        #         self.ComputedROIs.append(roi)

        # # Scale
        # result[:] *= self.ScalingFactor.value

        # # Add constant offset
        # result[:] += self.Offset.value
        
        # return result
        

    def computeBoxMean(self, xC,yC,zC):
        
        outVal = 0.0
        boxSize = 5
        # Set roi, making sure we don't access out of region
        if xC+boxSize > self.InputShape[0]:
            xEnd = self.InputShape[0]
        else:
            xEnd = xC+boxSize 

        if yC+boxSize > self.InputShape[1]:
            yEnd = self.InputShape[1]
        else:
            yEnd = yC+boxSize 

        if zC+boxSize > self.InputShape[2]:
            zEnd = self.InputShape[2]
        else:
            zEnd = zC+boxSize 

        for x in range(xC, xEnd):
            for y in range(yC, yEnd):
                for z in range(zC, zEnd):
                    outVal += self.CachedInput[x][y][z]
        return outVal/125 #5^3

    def computeResults(self, roi):
        for i in range(roi.start[0],roi.stop[0]):
            for j in range(roi.start[1],roi.stop[1]):
                for k in range(roi.start[2],roi.stop[2]):                    
                    self.CachedResults[i][j][k] = self.computeBoxMean(i,j,k)        
        return self.CachedResults[roi.start[0]:roi.stop[0], roi.start[1]:roi.stop[1], roi.start[2]:roi.stop[2],roi.start[3]:roi.stop[3]]

    def getCachedResults(self, roi):
        return self.CachedResults[roi.start[0]:roi.stop[0], roi.start[1]:roi.stop[1], roi.start[2]:roi.stop[2],roi.start[3]:roi.stop[3]]

    def propagateDirty(self, slot, subindex, roi):
        # If the dirty slot is one of our two constants, then the entire image region is dirty
        if slot == self.Offset or slot == self.ScalingFactor or slot == self.GaussianKernelSize:
            roi = slice(None) # The whole image region
        
        # All inputs affect all outputs, so every image is dirty now
        for oslot in self.Output:
            oslot.setDirty( roi )

    #############################################
    ## Methods to satisfy MultiLaneOperatorABC ##
    #############################################

    def addLane(self, laneIndex):
        """
        Add an image lane to the top-level operator.
        """
        numLanes = len(self.Input)
        assert numLanes == laneIndex, "Image lanes must be appended."        
        self.Input.resize(numLanes+1)
        self.Output.resize(numLanes+1)
        
    def removeLane(self, laneIndex, finalLength):
        """
        Remove the specified image lane from the top-level operator.
        """
        self.Input.removeSlot(laneIndex, finalLength)
        self.Output.removeSlot(laneIndex, finalLength)

    def getLane(self, laneIndex):
        return OperatorSubView(self, laneIndex)

assert issubclass(OpDummy, MultiLaneOperatorABC)



if __name__ == "__main__":
    from lazyflow.graph import Graph
    op = OpDummy(graph=Graph())

    shape = (5,5)
    zeros = np.zeros( shape, dtype=np.float32 )
    ones = np.ones( shape, dtype=np.float32 )
    twos = 2*np.ones( shape, dtype=np.float32 )

    scalingFactor = 5
    offset = 10
    
    op.ScalingFactor.setValue(scalingFactor)
    op.Offset.setValue(offset)
        
    op.Input.resize(3)
    op.Input[0].setValue( zeros )
    op.Input[1].setValue( ones )
    op.Input[2].setValue( twos )
    
    expected = offset + scalingFactor * (ones - (zeros + ones + twos) / len(op.Input)) 
    print "expected:", expected

    output = op.Output[1][:].wait()
    print "output:",output
    assert ( output == expected).all()
