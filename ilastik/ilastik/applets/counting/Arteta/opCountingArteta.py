#Python
import copy
from functools import partial
import itertools
import math
import time
import traceback 

#SciPy
import numpy
import vigra

#lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpValueCache, \
							   OpArrayCache, OpMultiArraySlicer2, \
							   OpPrecomputedInput, OpPixelOperator, OpMaxChannelIndicatorOperator, \
							   OpReorderAxes
from lazyflow.operators.opDenseLabelArray import OpDenseLabelArray

from lazyflow.request import Request, RequestPool
from lazyflow.roi     import roiToSlice, sliceToRoi, nonzero_bounding_box
							   
from ilastik.applets.counting.countingOperators import OpTrainCounter, OpPredictCounter, OpLabelPreviewer
from ilastik.applets.counting.opCounting import OpLabelPipeline, OpPredictionPipelineNoCache, \
												OpUpperBound, OpMean, OpBoxViewer, OpEnsembleMargin, \
												OpVolumeOperator
#ilastik

from ilastik.utility.operatorSubView import OperatorSubView
from ilastik.utility import OpMultiLaneWrapper
import threading
from ilastik.applets.base.applet import DatasetConstraintError

# Arteta
from ..Arteta.artetaInnerPipeline import ArtetaPipeline

class OpCountingArteta( Operator ):
	"""
	Top-level operator for counting
	"""
	name="OpCountingArteta"
	category = "Top-level"
	
	# Graph inputs
	
	InputImages = InputSlot(level=1) # Original input data.  Used for display only.

	LabelInputs = InputSlot(optional = True, level=1) # Input for providing label data from an external source
	BoxLabelInputs = InputSlot(optional = True, level=1) # Input for providing label data from an external source
	LabelsAllowedFlags = InputSlot(stype='bool', level=1) # Specifies which images are permitted to be labeled 
	FeatureImages = InputSlot(level=1) # Computed feature images (each channel is a different feature)
	CachedFeatureImages = InputSlot(level=1) # Cached feature data.
	FreezePredictions = InputSlot(stype='bool', value=False)
	PredictionsFromDisk = InputSlot(optional=True, level=1)
	PredictionProbabilities = OutputSlot(level=1) # Classification predictions (via feature cache for interactive speed)

	#PredictionProbabilityChannels = OutputSlot(level=2) # Classification predictions, enumerated by channel
	#SegmentationChannels = OutputSlot(level=2) # Binary image of the final selections.
	
	LabelImages = OutputSlot(level=1) # Labels from the user
	BoxLabelImages= OutputSlot(level=1) # Input for providing label data from an external source
	NonzeroLabelBlocks = OutputSlot(level=1) # A list if slices that contain non-zero label values
	Classifier = OutputSlot() # We provide the classifier as an external output for other applets to use
	CachedPredictionProbabilities = OutputSlot(level=1) # Classification predictions (via feature cache AND prediction cache)
	HeadlessPredictionProbabilities = OutputSlot(level=1) # Classification predictions ( via no image caches (except for the classifier itself )

	UncertaintyEstimate = OutputSlot(level=1)

	MaxLabelValue = OutputSlot()

	# GUI-only (not part of the pipeline, but saved to the project)
	UpperBound = OutputSlot()
	LabelNames = OutputSlot()
	LabelColors = OutputSlot()
	PmapColors = OutputSlot()
	Density = OutputSlot(level=1)
	LabelPreview = OutputSlot(level=1)
	OutputSum = OutputSlot(level=1)

	def __init__( self, *args, **kwargs ):
		"""
		Instantiate all internal operators and connect them together.
		"""
		super(OpCountingArteta, self).__init__(*args, **kwargs)
		
		# Default values for some input slots
		self.FreezePredictions.setValue(True)
		self.LabelNames.setValue( ["Objects", "Background"] )
		self.LabelColors.setValue( [ (255,0,0), (0,255,0) ] )
		self.PmapColors.setValue( [ (255,0,0), (0,255,0) ] )

		# SPECIAL connection: The LabelInputs slot doesn't get it's data  
		#  from the InputImages slot, but it's shape must match. (inputs are from GUI)
		self.LabelInputs.connect( self.InputImages )
		self.BoxLabelInputs.connect( self.InputImages )

		# Hook up Labeling Pipeline
		self.opLabelPipeline = OpMultiLaneWrapper( OpLabelPipeline, parent=self )
		self.opLabelPipeline.RawImage.connect( self.InputImages )
		self.opLabelPipeline.LabelInput.connect( self.LabelInputs )
		self.opLabelPipeline.BoxLabelInput.connect( self.BoxLabelInputs )
		self.LabelImages.connect( self.opLabelPipeline.Output )
		self.NonzeroLabelBlocks.connect( self.opLabelPipeline.nonzeroBlocks )
		self.BoxLabelImages.connect( self.opLabelPipeline.BoxOutput)

		self.GetFore= OpMultiLaneWrapper(OpPixelOperator,parent = self) # gets objects only
		def conv(arr):
			numpy.place(arr, arr ==2, 0)
			return arr.astype(numpy.float)
		self.GetFore.Function.setValue(conv)
		self.GetFore.Input.connect(self.opLabelPipeline.Output)

		self.LabelPreviewer = OpMultiLaneWrapper(OpLabelPreviewer, parent = self)
		self.LabelPreviewer.Input.connect(self.GetFore.Output)
		self.LabelPreview.connect(self.LabelPreviewer.Output)

		# Hook up the Training operator
		self.opUpperBound = OpUpperBound( parent= self, graph= self.graph )
		self.UpperBound.connect(self.opUpperBound.UpperBound)

		self.boxViewer = OpBoxViewer( parent = self, graph=self.graph )

		self.opTrain = OpTrainArtetaCounter( parent=self, graph=self.graph )
		self.opTrain.inputs['InputImages'].connect( self.InputImages)
		self.opTrain.inputs['Labels'].connect( self.GetFore.Output) 
		# FIXME : it should be done using the same 'slot' mechanism, this is done "manually", through GUI interface, thus the next line is commented.
		# self.opTrain.inputs['BoxConstraintValues'].connect( self.opLabelPipeline.BoxOutput ) 
		self.opTrain.inputs['Features'].connect( self.FeatureImages ) 
		self.opTrain.inputs["nonzeroLabelBlocks"].connect( self.opLabelPipeline.nonzeroBlocks )
		self.opTrain.inputs['fixClassifier'].setValue( True )

		# Hook up the Classifier Cache
		# The classifier is cached here to allow serializers to force in
		#   a pre-calculated classifier (loaded from disk)
		self.classifier_cache = OpValueCache( parent=self, graph=self.graph )
		self.classifier_cache.inputs["Input"].connect(self.opTrain.outputs['Classifier'])
		self.Classifier.connect( self.classifier_cache.Output )

		# Hook up the prediction pipeline inputs 
		self.opPredictionPipeline = OpMultiLaneWrapper( OpArtetaPredictionPipeline, parent=self )
		self.opPredictionPipeline.InputImages.connect( self.InputImages )
		self.opPredictionPipeline.FeatureImages.connect( self.FeatureImages )
		self.opPredictionPipeline.CachedFeatureImages.connect( self.CachedFeatureImages )
		self.opPredictionPipeline.Classifier.connect( self.classifier_cache.Output )
		self.opPredictionPipeline.FreezePredictions.connect( self.FreezePredictions )
		self.opPredictionPipeline.PredictionsFromDisk.connect( self.PredictionsFromDisk )

		# Prediction pipeline outputs -> Top-level outputs
		self.PredictionProbabilities.connect( self.opPredictionPipeline.PredictionProbabilities )
		self.CachedPredictionProbabilities.connect( self.opPredictionPipeline.CachedPredictionProbabilities )
		self.HeadlessPredictionProbabilities.connect( self.opPredictionPipeline.HeadlessPredictionProbabilities )
		self.UncertaintyEstimate.connect( self.opPredictionPipeline.UncertaintyEstimate )
		self.Density.connect(self.opPredictionPipeline.CachedPredictionProbabilities ) 
		self.OutputSum.connect(self.opPredictionPipeline.OutputSum )

		def inputResizeHandler( slot, oldsize, newsize ):
			if ( newsize == 0 ):
				self.LabelImages.resize(0)
				self.NonzeroLabelBlocks.resize(0)
				self.PredictionProbabilities.resize(0)
				self.CachedPredictionProbabilities.resize(0)
		self.InputImages.notifyResized( inputResizeHandler )

		# Debug assertions: Check to make sure the non-wrapped operators stayed that way.
		assert self.opTrain.Features.operator == self.opTrain

		def handleNewInputImage( multislot, index, *args ):
			def handleInputReady(slot):
				# Chris - maybe a really bad idea - check constraints now?
				self._checkConstraints(index)
				self.setupCaches( multislot.index(slot) )
			multislot[index].notifyReady(handleInputReady)
				
		self.InputImages.notifyInserted( handleNewInputImage )

		# All input multi-slots should be kept in sync
		# Output multi-slots will auto-sync via the graph
		multiInputs = filter( lambda s: s.level >= 1, self.inputs.values() )
		for s1 in multiInputs:
			for s2 in multiInputs:
				if s1 != s2:
					def insertSlot( a, b, position, finalsize ):
						a.insertSlot(position, finalsize)
					s1.notifyInserted( partial(insertSlot, s2 ) )
					
					def removeSlot( a, b, position, finalsize ):
						a.removeSlot(position, finalsize)
					s1.notifyRemoved( partial(removeSlot, s2 ) )
		
		
		# workaround
		self.options = []

	def setupOutputs(self):
		self.LabelNames.meta.dtype = object
		self.LabelNames.meta.shape = (1,)
		self.LabelColors.meta.dtype = object
		self.LabelColors.meta.shape = (1,)
		self.PmapColors.meta.dtype = object
		self.PmapColors.meta.shape = (1,)
		self.MaxLabelValue.setValue(2)


	def setupCaches(self, imageIndex):
		numImages = len(self.InputImages)
		inputSlot = self.InputImages[imageIndex]
		self.LabelInputs.resize(numImages)
		self.BoxLabelInputs.resize(numImages)

		# Special case: We have to set up the shape of our label *input* according to our image input shape
		shapeList = list(self.InputImages[imageIndex].meta.shape)
		try:
			channelIndex = self.InputImages[imageIndex].meta.axistags.index('c')
			shapeList[channelIndex] = 1
		except:
			pass
		self.LabelInputs[imageIndex].meta.shape = tuple(shapeList)
		self.LabelInputs[imageIndex].meta.axistags = inputSlot.meta.axistags
		self.BoxLabelInputs[imageIndex].meta.shape = tuple(shapeList)
		self.BoxLabelInputs[imageIndex].meta.axistags = inputSlot.meta.axistags

	def setInSlot(self, slot, subindex, roi, value):
		# Nothing to do here: All inputs that support __setitem__
		#   are directly connected to internal operators.
		pass

	def propagateDirty(self, slot, subindex, roi):
		# Nothing to do here: All outputs are directly connected to 
		#  internal operators that handle their own dirty propagation.
		pass

	def addLane(self, laneIndex):
		numLanes = len(self.InputImages)
		assert numLanes == laneIndex, "Image lanes must be appended."        
		self.InputImages.resize(numLanes+1)
		self.opTrain.BoxConstraintRois.resize(numLanes + 1)
		self.opTrain.BoxConstraintValues.resize(numLanes + 1)
		self.boxViewer.rois.resize(numLanes + 1)
		
	def removeLane(self, laneIndex, finalLength):
		self.InputImages.removeSlot(laneIndex, finalLength)
		self.opTrain.BoxConstraintRois.removeSlot(laneIndex, finalLength)
		self.opTrain.BoxConstraintValues.removeSlot(laneIndex, finalLength)
		self.boxViewer.rois.removeSlot(laneIndex, finalLength)

	def getLane(self, laneIndex):
		return OperatorSubView(self, laneIndex)
	
	def _checkConstraints(self, laneIndex):
		"""
		Ensure that all input images have the same number of channels
		"""

		thisLaneTaggedShape = self.InputImages[laneIndex].meta.getTaggedShape()
		
		if self.InputImages[laneIndex].meta.getAxisKeys()[-1] != 'c':
			raise DatasetConstraintError(
				 "Objects Couting Workflow Counting",
				 "This code assumes channel is the last axis")

		validShape = thisLaneTaggedShape
		for i, slot in enumerate(self.InputImages):
			if slot.ready() and i != laneIndex:
				validShape = slot.meta.getTaggedShape()
				break
		
		if len(validShape) != len(thisLaneTaggedShape):
			raise DatasetConstraintError(
				 "Objects Couting Workflow Counting",
				 "All input images must have the same dimensionality.  "\
				 "Your new image has {} dimensions (including channel), but your other images have {} dimensions."\
				 .format( len(thisLaneTaggedShape), len(validShape) ) )
			
		if validShape['c'] != thisLaneTaggedShape['c']:
			raise DatasetConstraintError(
				 "Objects Counting Workflow",
				 "All input images must have the same number of channels.  "\
				 "Your new image has {} channel(s), but your other images have {} channel(s)."\
				 .format( thisLaneTaggedShape['c'], validShape['c'] ) )

class OpArtetaPredictionPipelineNoCache(Operator):
	"""
	This contains only the cacheless parts of the prediction pipeline, for easy use in headless workflows.
	"""
	InputImages = InputSlot()
	FeatureImages = InputSlot()
	# MaxLabel = InputSlot() # this slot is never set --> makes our Operator.Density not ready --> helps us, bug at launching otherwise --> why?
	Classifier = InputSlot()
	FreezePredictions = InputSlot()
	PredictionsFromDisk = InputSlot( optional=True )
	
	HeadlessPredictionProbabilities = OutputSlot() # drange is 0.0 to 1.0
	
	OutputSum = OutputSlot()

	def __init__(self, *args, **kwargs):
		super( OpArtetaPredictionPipelineNoCache, self ).__init__( *args, **kwargs )

		self.cacheless_predict = OpPredictArtetaCounter( parent=self )
		self.cacheless_predict.name = "OpPredictCounter (Cacheless Path)"
		self.cacheless_predict.inputs['Classifier'].connect(self.Classifier) 
		self.cacheless_predict.inputs['Features'].connect(self.FeatureImages) # <--- Not from cache
		self.cacheless_predict.inputs['Image'].connect(self.InputImages) # <--- Not from cache
		# 
		# self.cacheless_predict.inputs['LabelsCount'].connect(self.MaxLabel)
		self.meaner = OpMean(parent = self)
		self.meaner.Input.connect(self.cacheless_predict.PMaps)
		self.HeadlessPredictionProbabilities.connect(self.meaner.Output)

		self.opVolumeSum = OpVolumeOperator(parent=self)
		self.opVolumeSum.Input.connect(self.meaner.Output)
		self.opVolumeSum.Function.setValue(numpy.sum)

		self.OutputSum.connect( self.opVolumeSum.Output )


	def setupOutputs(self):
		pass

	def execute(self, slot, subindex, roi, result):
		assert False, "Shouldn't get here.  Output is assigned a value in setupOutputs()"

	def propagateDirty(self, slot, subindex, roi):
		# Our output changes when the input changed shape, not when it becomes dirty.
		pass

class OpArtetaPredictionPipeline(OpArtetaPredictionPipelineNoCache):
	"""
	This operator extends the cacheless prediction pipeline above with additional outputs for the GUI.
	(It uses caches for these outputs, and has an extra input for cached features.)
	"""        
	CachedFeatureImages = InputSlot()

	PredictionProbabilities = OutputSlot()
	CachedPredictionProbabilities = OutputSlot()
	UncertaintyEstimate = OutputSlot()

	def __init__(self, *args, **kwargs):
		super(OpArtetaPredictionPipeline, self).__init__( *args, **kwargs )

		# Prediction using CACHED features.
		self.predict = OpPredictArtetaCounter( parent=self )
		self.predict.name = "OpPredictArtetaCounter"
		self.predict.inputs['Classifier'].connect(self.Classifier) 
		self.predict.inputs['Image'].connect(self.InputImages)
		self.predict.inputs['Features'].connect(self.CachedFeatureImages)
		self.PredictionProbabilities.connect( self.predict.PMaps )

		# Prediction cache for the GUI
		self.prediction_cache_gui = OpArrayCache( parent=self )
		self.prediction_cache_gui.name = "prediction_cache_gui"
		self.prediction_cache_gui.inputs["fixAtCurrent"].connect( self.FreezePredictions )
		self.prediction_cache_gui.inputs["Input"].connect( self.predict.PMaps )
		
		# Also provide each prediction channel as a separate layer (for the GUI) 
		# Uncertainty not used yet.
		self.opUncertaintyEstimator = OpEnsembleMargin( parent=self )
		self.opUncertaintyEstimator.Input.connect( self.prediction_cache_gui.Output )

		## Cache the uncertainty so we get zeros for uncomputed points
		self.opUncertaintyCache = OpArrayCache( parent=self )
		self.opUncertaintyCache.name = "opUncertaintyCache"
		self.opUncertaintyCache.blockShape.setValue(self.FeatureImages.meta.shape)
		self.opUncertaintyCache.Input.connect( self.opUncertaintyEstimator.Output )
		self.opUncertaintyCache.fixAtCurrent.connect( self.FreezePredictions )
		self.UncertaintyEstimate.connect( self.opUncertaintyCache.Output )
		
		self.meaner = OpMean(parent = self)
		self.meaner.Input.connect(self.prediction_cache_gui.Output)

		self.precomputed_predictions_gui = OpPrecomputedInput( parent=self )
		self.precomputed_predictions_gui.name = "precomputed_predictions_gui"
		self.precomputed_predictions_gui.SlowInput.connect( self.meaner.Output )
		self.precomputed_predictions_gui.PrecomputedInput.connect( self.PredictionsFromDisk )
		self.CachedPredictionProbabilities.connect(self.precomputed_predictions_gui.Output)

	def setupOutputs(self):
		# set cache block shape to input dimension
		self.prediction_cache_gui.blockShape.setValue(self.predict.PMaps.meta.shape)
		# pass

class OpPredictArtetaCounter(Operator):
	name = "PredictArtetaCounter"
	description = "Predict on multiple images, with Arteta objects counting implementation"
	category = "Learning"

	inputSlots = [	InputSlot("Image"), 
					InputSlot("Classifier"),
					InputSlot("Features")]

	outputSlots = [	OutputSlot("PMaps"),
					OutputSlot("OutputSum")] 
	
	def __init__(self, *args, **kwargs):
		super(OpPredictArtetaCounter, self).__init__( *args, **kwargs )
		# we have to set output as dirty 
		self.outputs["PMaps"].setDirty()

	def setupOutputs(self):
		self.PMaps.meta.dtype = numpy.float32
		self.PMaps.meta.axistags = copy.copy(self.Image.meta.axistags)
		# Channel is the last axis, set it to 1 for output
		self.PMaps.meta.shape = (self.Image.meta.shape[:-1] + (1,))
		self.PMaps.meta.drange = (0, 1.0)

	def execute(self, slot, subindex, roi, result):
		print '[OpPredictArtetaCounter] - computing count predictions for roi ', roi
		
		classifier =self.inputs["Classifier"][:].wait()
		feats = self.inputs["Features"][:].wait()
		mask = numpy.ones((feats.shape[:-1]  ),dtype=bool) # + (1,) # removed, useless

		if classifier is None:
			# Training operator may return 'None' if there was no data to train with
			print '[OpPredictArtetaCounter] - No classifier supplied, returning zeros'
			return numpy.zeros(numpy.subtract(roi.stop, roi.start), dtype=numpy.float32)[...]

		# hack to add a useless axis to mask and feats, so that they work with predict (using map)
		isSingleImage = False
		if len(mask.shape) == 2 : 
			mask = mask[None,...]
			feats = feats[None, ...]
			isSingleImage = True

		# actual prediction 
		res = classifier[0].predict(feats,mask) 
		result = numpy.asarray(res)[...,None] # add channel axis 

		# return value 
		res = result
		roiAsSlice = roiToSlice(roi.start,roi.stop)
		print 'RoiAsSlice : ', roiAsSlice
		print 'Result global shape ', res.shape
		if isSingleImage : 
			res = res[0,...] # remove t dimension, as it wasn't in the input meta

		return res[roiAsSlice]

	def propagateDirty(self, slot, subindex, roi):
		self.outputs["PMaps"].setDirty()



class OpTrainArtetaCounter(Operator):
	name= "TrainArtetaClassifier"
	description = "Train a counter based on Lempitsky-Arteta paper Interactive Object Counting"
	category = "Learning"

	inputSlots = [InputSlot("InputImages", level=1),
				  InputSlot("Features", level=1),
				  InputSlot("Labels", level=1), 
				  InputSlot("fixClassifier", stype="bool",optional=True),
				  InputSlot("nonzeroLabelBlocks", level=1,optional =True),
				  InputSlot("Sigma", stype = "float", value=2.0), # Gaussian sigma
				  InputSlot("MaxDepth", stype = "int", value=4), #KDTree parameter
				  InputSlot("BoxConstraintRois", level = 1, stype = "list", value = [], optional=True),
				  InputSlot("BoxConstraintValues", level = 1, stype = "list", value = [], optional=True),
				  InputSlot("BoxesCoords", stype = "list", value = [], optional=True)
				 ]
	outputSlots = [OutputSlot("Classifier")]

	def __init__(self,*args,**kwargs):
		super(OpTrainArtetaCounter,self).__init__(*args,**kwargs)
		self.arteta_pipeline = ArtetaPipeline()
		self.Classifier.meta.dtype = object
		self.Classifier.meta.shape = (1,)

		self.coords = []
		# default values
		self.fixClassifier.setValue(False) 
		self.MaxDepth.setValue(4)
		self.Sigma.setValue(3.0)

		self.inputs["Features"].meta.dtype = numpy.ndarray

	def setupOutputs(self):
		# if(self.inputs["fixClassifier"]).value == False:
		# 	params = {"sigma": self.Sigma.value,"maxDepth" : self.MaxDepth.value}
		# 	self.arteta_pipeline.set_params(**params)
		pass
		
	def propagateDirty(self, slot, subindex, roi):
		if slot is not self.inputs["fixClassifier"] and self.inputs["fixClassifier"].value == False:
			self.outputs["Classifier"].setDirty()

	def execute(self, slot, subindex, roi, result):

		print '[OpTrainArtetaCounter] - training classifier'
		# read inputs
		feats = self.Features[0][:].wait() 
		labels = self.Labels[0][:].wait()
		imgs = self.InputImages[0][:].wait()
		boxes = self.BoxesCoords[:].wait()

		print self.Labels[0].meta.axistags
		print self.InputImages[0].meta.axistags
		print self.Features[0].meta.axistags

		# set params from UI
		params = {"sigma": self.Sigma.value,"maxDepth" : self.MaxDepth.value}
		self.arteta_pipeline.set_params(**params)

		# do we have a single image or a serie along time ?
		if self.Labels[0].meta.axistags.index('t') < len(self.Labels[0].meta.axistags) : 
			# getting annotations coordinates
			labels = self.Labels[0][:].wait()
			labels_coordinates = numpy.nonzero(labels)

			# getting images (along t dimension) that have annotations
			assert self.Labels[0].meta.axistags.index('t') == 0, "t axis should be at position 0, not {}".format(self.Labels[0].meta.axistags.index('t'))
			annotated_frames = numpy.unique(labels_coordinates[0]) 
			print 'T values where there are annotations : ', annotated_frames

			# extracting images with annotations
			annotated_imgs = imgs[annotated_frames]
			annotated_feats = feats[annotated_frames]
			annotated_layers = labels[annotated_frames]
			
			# FIXME : locate t position of masks
			# compute mask based on boxes
			mask = numpy.zeros((annotated_imgs.shape ),dtype=bool)
			mask = self.computeMask(mask,boxes[0], invertXY=True)

			# train estimator
			self.arteta_pipeline.fit(annotated_imgs, annotated_feats,annotated_layers,mask[...,0],True)

		else:
			# compute mask based on boxes
			mask = numpy.zeros((labels.shape ),dtype=bool)
			# add t axis at the beginning of the mask, to have the same shape length as multiple arrays.
			mask = mask[None,...] 
			mask = self.computeMask(mask,boxes[0])

			# train estimator
			self.arteta_pipeline.fit(imgs[...,0], feats,labels[...,0],mask[0,...,0])
		
		result = self.arteta_pipeline
		return [result]


	# def setCoords(self,coords):
	# 	print 'Coords are : ', coords
	# 	if(self.coords != coords):
	# 		self.coords = coords
	# 		self.outputs['Classifier'].setDirty() # boxes modified, output is dirty

	def computeMask(self,mask, boxes,invertXY=False): 
	# TODO : locate t position of masks
	# if two boxes overlap, nothing special happens, 
	#	the overlapped area will be taken into account once.
	
		for box in boxes:
			x = box['x']
			y = box['y']
			w = box['w']
			h = box['h']
			# print 'Box dimensions : ', h, w
			if(invertXY):
				mask[:,y:y+h, x:x+w] = 1 
			else:
				mask[:, x:x+w,y:y+h] = 1 
		return mask


	def checkInputsReady(self):
		for item in self.inputs :
			print 'Input ' , item , ' ready state is : ', self.inputs[item].ready()
		print 'Output dirty state : ', self.outputs["Classifier"].dirty


