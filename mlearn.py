#  This is a module containing the following classifiers:
#  - Naive (Gauss) Bayes Classifier 
#  - Binary Logistic Regression 
#  - AdaBoosted decision stump
#
#  by Shih-Ho Cheng (shihho.cheng@gmail.com)

from pylab import *
import sys
import scipy.stats as st

class Classifier:
  """Base class for all the classifiers
    
    Args:
      trainingData: training data set
      target: target of the training data set

    Attributes:
      tData: the training data itself
      tTarget: the target of tData
      classes: an array containing the distinct classes
      nTData: number of training examples
      nFeatures: number of features (dimensions) in the data
      nClasses: number of (distinct) classes or classifications
      hasTrained: flag indicating whether the object has been trained
      isCrossValidated: flag indicating whether the object has been cross-validated

    Methods:
      train: template method for the daughters classes (specific learners)
      classify: template method for the daughters classes (specific learners) 
      crossValidate: cross validates the training given a test data set
  """

  def __init__(self, trainingData, target):
    """Initializes the Classifier objects and populates their attributes"""
    if trainingData.shape[0]==len(target):
      self.tData   = trainingData                   # training data array
      self.tTarget = target                         # target class array
      self.classes = array( list(set(target)) )     # (non-repeated) class array 
      self.nTData  = trainingData.shape[0]          # N of training samples
      self.nFeatures = trainingData.shape[1]        # N of features per sample
      self.nClasses = len(self.classes)             # N of classes (non-repeated) classes
      self.hasTrained = False 
      self.isCrossValidated = False
    else:
      print "   <!> ERROR: length of training sample doesn't match the length \
                           of target array!"
      sys.exit(1)

  def train():
    """Dummy method for the inherited classes"""
    pass

  def classify():
    """Dummy method for the inherited classes"""
    pass

  def crossValidate(self, testData, testTarget):
    """Creates a cross validation table with a test Data set """
    if self.hasTrained:
      if len(testData)==len(testTarget) and set(testTarget)==set(self.classes):
        # Create and initialize the crossValMatrix dictionary (of a dictionary)
        crossValMatrix = {}
        for ci in self.classes:
          crossValMatrix[ci] = {}
          for cj in self.classes:
            crossValMatrix[ci][cj] = 0
        testClassification = self.classify(testData)
        for i in range(len(testTarget)):
          crossValMatrix[testTarget[i]][testClassification[i]] += 1
        self.crossValidated = True
        return crossValMatrix
      else:
        print "   <!> WARNING: the test data and test target don't have the same dimensions."
    else:
      print "   <!> WARNING: The learner has not been trained yet."
  

class Ngbayes(Classifier):
  """Naive gaussian bayes classifier for 
     continuous feature and discrete targets.
  """

  def train(self):
    """Trains the Ngbayes object"""
    # ML estimate of sample mean and sigma for each feature of each classification
    self.mean  = zeros( (len(self.classes),self.nFeatures) )
    self.sigma = zeros( (len(self.classes),self.nFeatures) )
    for c in range(len(self.classes)):              # find the samples belonging to class c
      sampleStatus = (self.tTarget==self.classes[c])   # True if t==0, False otherwise
      buffer_tData = self.tData[sampleStatus,0:] 
      num_c_Samples = len(buffer_tData) # Number of samples with classification c
      self.mean[c,0:]  = buffer_tData.sum(axis=0)/(num_c_Samples*1.)
      self.sigma[c,0:] = ((buffer_tData - self.mean[c,0:])**2).sum(axis=0)/(num_c_Samples-1.)
    self.hasTrained = True

  def classify(self, data):
    """Runs the input data forward through the trained
       classifier and returns the predicted classification
    """
    if self.hasTrained:
      classification = zeros(len(data))
      for i in range(len(data)):
        dataLine   = repeat( data[i,0:].reshape(1,len(data[i,0:])), len(self.classes), axis=0 )
        condProb   = st.norm.pdf(dataLine, self.mean, self.sigma)    
        TTcondProb = condProb.prod(axis=1)
        max = 0
        #candPosterioProb = zeros(len(self.nClasses))
        for c in range(len(self.classes)):
          priorProb = list(self.tTarget).count(self.classes[c]) / (len(self.tTarget)*1.)
          posteriorProb  = TTcondProb[c]*priorProb
          if posteriorProb>=max:
            max = posteriorProb
            classification[i] = self.classes[c]
      return classification
    else:
      print "<!> WARNING: the Ngbayes classifier hasn't been trained yet."


class BinLogReg(Classifier):
  """Implementation of a binary logistic regression classifier
     (the classification of training samples must be 0 or 1)
  """

  def train(self, eta=0.001, numIter=1000):
    """Trains the logistic regression"""
    if set(self.classes)==set([0,1]):
      buffer_data = concatenate( (ones((self.nTData,1)),self.tData), axis=1 )
      w = zeros( self.nFeatures+1 )
      for i in range(numIter):
        expArg = repeat(w[0],self.nTData) + sum( w[1:]*buffer_data[:,1:], axis=1 ) 
        estimatedP  = exp(expArg)/(1.+exp(expArg))
        gradientDir = sum( transpose(buffer_data)*(self.tTarget - estimatedP), axis=1 )
        w = w + eta*gradientDir
      self.hasTrained = True
      self.w = w
    else:
      print "<!> WARNING: Unable to train with logit reg! \
                          The binary classes needs to be (0,1)." 
  
  def classify(self, data):
    """Classifies with the trained classifier"""
    if self.hasTrained:
      inferredClass = self.w[0] + sum( self.w[1:]*data, axis=1 )
      decision = (inferredClass<0)
      decision = [0 if i else 1 for i in decision]
      return array(decision)
    else:
      print "<!> WARNING: the LogReg classifier hasn't been trained yet."


class Stump(Classifier):
  """A decision stump
    This is a weak classifier by itself. It should be use in conjunction
    with the adaBoost meta-algorithm.
    NOTE: all misclassification weights are considered to be the same,
    i.e., misclassifying c1 as c2 carries the same penalty as misclassifying
    c1 as c3.
  """ 

  def __mostCommon(self, classes):
    """Finds the most common element in an array of classes"""
    mostCommon = classes[random_integers(0,len(classes)-1)]
    maxCount   = 0
    for c in classes:
      count = list(classes).count(c)
      if count > maxCount:
        mostCommon = c
        maxCount   = count
      else:
        continue
    return mostCommon

  def train(self, weights):
    candBoundary = zeros(self.nFeatures)   # cand. split boundaries per feature
    candError     = zeros(self.nFeatures)  # gini coef. at the candidate split
    candLeftClass= zeros(self.nFeatures)   # cand. class that satisfies the split condition (<)
    candRightClass= zeros(self.nFeatures)  # cand. class that satisfies the split condition (<)
    for feature in range(self.nFeatures):
      # sorted tData indices with 'feature'
      idx_sorted = lexsort( (self.tTarget,self.tData[:,feature]) )
      buffer_tData   = self.tData[idx_sorted,feature] # sorted copy of tData
      buffer_target  = self.tTarget[idx_sorted]        # sorted (according to tData) copy of target
      buffer_weights = weights[idx_sorted]            # sorted copy of weights
      # find the indices of candidate splits
      candStatus = diff(buffer_target)!=0       # this has one unit length less than tData
      critIdx    = where(candStatus)[0]         # get the indices where candStatus is True
      errAtBoundaries   = []
      boundaries        = []
      classesToTheLeft  = [] # majority of classes satisfying the split condition (< boundary)
      classesToTheRight = [] # majority of classes NOT satisfying the split condition (> boundary)
      for i in critIdx:
        critLimit = (buffer_tData[i+1]+buffer_tData[i])/2.
        boundaries.append( critLimit )  
        leftClass  = self.__mostCommon(buffer_target[:i+1])
        # This just gives the complementary of leftClass
        rightClass = self.classes[self.classes!=leftClass][0] 
        classesToTheLeft.append( leftClass )
        classesToTheRight.append( rightClass )
        classError  = sum( buffer_weights[:i+1] * (buffer_target[:i+1]!=leftClass) )
        classError += sum( buffer_weights[i+1:] * (buffer_target[i+1:]!=rightClass) )
        classError /= self.nTData*1.
        errAtBoundaries.append( classError )
      candError[feature]       = min(errAtBoundaries)
      candBoundary[feature]   = boundaries[ argmin(errAtBoundaries) ]
      candLeftClass[feature]  = classesToTheLeft[ argmin(errAtBoundaries) ]
      candRightClass[feature] = classesToTheRight[ argmin(errAtBoundaries) ]
    self.classError = min(errAtBoundaries)
    self.rootBoundary = candBoundary[ argmin(candError) ]
    self.rootFeature  = argmin(candError)
    self.leftClass    = candLeftClass[ argmin(candError) ]
    self.rightClass   = candRightClass[ argmin(candError) ]
    self.hasTrained   = True

  def classify(self, data):
    if self.hasTrained:
      classification = zeros(len(data))
      for i in range(len(data)):
        if data[i,self.rootFeature]<=self.rootBoundary:
          classification[i] = self.leftClass
        else:
          classification[i] = self.rightClass
      return classification
    else:
      print "<!> WARNING: the Stump classifier hasn't been trained yet."
  

class adaBoostStump(Stump):
  """Binary adaptive boosting implementation of 
     the stump decision.
     Note: The classification needs to be -1 or 1.
  """
  
  def bTrain(self, maxNumIter=500, errorTolerance=1e-6, verbose=True):
    """Trains the decision stump with adaboost"""
    # Initialize the weights and the ensemble weight (alpha) holder
    weights  = ones(self.nTData)/(self.nTData*1.)
    alphaErr = []
    classFeature = []
    classCritLim = []
    classOnLeft  = []
    classOnRight  = []
    for i in range(maxNumIter):
      if verbose:
        print ">> Iter", i
      self.train(weights)
      inferredClass = self.classify(self.tData)
      alpha = 0.5*log((1.-self.classError)/self.classError)
      alphaErr.append( alpha )   # collect the ensemble weights
      classFeature.append( self.rootFeature )
      classCritLim.append( self.rootBoundary )
      classOnLeft.append( self.leftClass )
      classOnRight.append( self.rightClass )
      if verbose:
        print "Nmisclassified=", sum(inferredClass!=self.tTarget)
        print "err=", self.classError, "alpha=", alpha
        print "left=", self.leftClass, "right=", self.rightClass, \
              "rootBound=", self.rootBoundary, "rootFeat=", self.rootFeature
        print "--"
        print 
      # Update the weights
      idx = where(inferredClass!=self.tTarget)[0]
      weights[idx] = weights[idx] * exp( alpha )
      idx = where(inferredClass==self.tTarget)[0]
      weights[idx] = weights[idx] * exp( -alpha )
      weights = weights/sum(weights)
      if self.classError<errorTolerance: break
      else: continue
    self.alphaErr = array(alphaErr)
    self.adaBFeature = array(classFeature)
    self.adaBCritLim = array(classCritLim)
    self.adaBOnLeft  = array(classOnLeft)
    self.adaBOnRight = array(classOnRight)
    self.hasTrained = True

  def bClassify(self, data):
    """Returns the ensemble decision"""
    if self.hasTrained:
      accSum = zeros(len(data))
      for i in range(len(data)):
        for f in range(len(self.adaBFeature)):
          if data[i,self.adaBFeature[f]]<=self.adaBCritLim[f]:
            accSum[i] += self.adaBOnLeft[f]*self.alphaErr[f]
          else:
            accSum[i] += self.adaBOnRight[f]*self.alphaErr[f]
      return sign(accSum)
    else:
      print "   <!> WARNING: adaboost has not been trained yet"

  def bCrossValidate(self, testData, testTarget):
    """Creates a cross validation table for the adaBoost
       example with a test Data set and their true classifications
    """
    if self.hasTrained:
      if len(testData)==len(testTarget) and set(testTarget)==set(self.classes):
        # Create and initialize the crossValMatrix dictionary (of a dictionary)
        bCrossValMatrix = {}
        for ci in self.classes:
          bCrossValMatrix[ci] = {}
          for cj in self.classes:
            bCrossValMatrix[ci][cj] = 0
        testClassification = self.bClassify(testData)
        for i in range(len(testTarget)):
          bCrossValMatrix[testTarget[i]][testClassification[i]] += 1
        return bCrossValMatrix
      else:
        print "   <!> WARNING: the test data and test target don't have the same dimensions."
    else:
      print "   <!> WARNING: adaboost has not been trained yet"
