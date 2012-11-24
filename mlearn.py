"""
  This is a module containing the following classifiers:
  - Naive Gauss Bayes Classifier 
  - Logistic Regression 
  - Decision stump

  by Shih-Ho Cheng (shihho.cheng@gmail.com)
"""

from pylab import *
import sys
import scipy.stats as st

class Classifier:

  """
    Base class for all the classifiers
  """

  def __init__(self, trainingData, target):
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
      print "   <!> ERROR: length of training sample doesn't match the length of target array!"
      sys.exit(1)

  def crossValidate(self, testData, testTarget):
    """
      Creates a cross validation table with a test Data set and their 
      true classifications
    """
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
  

class Ngbayes(Classifier):

  """
    Naive gaussian bayes classifier for continuous feature and discrete targets.
  """

  def train(self):
    """
      Trains the Ngbayes object
    """
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
    """
      Runs the input data forward through the trained
      classifier and returns the predicted classification
      NOTE: the for loops could probably still be vectorized!
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


class SimpleLogReg(Classifier):

  """
    Implementation of a simple two-class logistic regression classifier
    (the classification of training samples must be 0 or 1)
  """

  def train(self, eta=0.001, numIter=1000):
    """
      Trains the logistic regression
    """
    #w = st.uniform.rvs( 0.001,0.5,size=self.nFeatures+1 )   # Initialize the weights to zero (with additional w_o)
    if set(self.classes)==set([-1,1]):
    buffer_data = concatenate( (ones((self.nTData,1)),self.tData), axis=1 )
    w = zeros( self.nFeatures+1 )
    for i in range(numIter):
      expArg = repeat(w[0],self.nTData) + sum( w[1:]*buffer_data[:,1:], axis=1 ) 
      estimatedP  = exp(expArg)/(1.+exp(expArg))
      gradientDir = sum( transpose(buffer_data)*(self.tTarget - estimatedP), axis=1 )
      w = w + eta*gradientDir
    self.hasTrained = True
    self.w = w
  
  def classify(self, data):
    """
      Classifies with the trained classifier
    """
    if self.hasTrained:
      #print self.w
      inferredClass = self.w[0] + sum( self.w[1:]*data, axis=1 )
      decision = (inferredClass<0)
      #print inferredClass
      decision = [0 if i else 1 for i in decision]
      return array(decision)
    else:
      print "<!> WARNING: the LogReg classifier hasn't been trained yet."
