#!/usr/bin/python
  
#
# Test driver for the mlearn module
# by Shih-Ho Cheng (shihho.cheng@gmail.com)
#

from pylab import *
import scipy.stats as st
import mlearn

# Create a binary class data set
nPts  = 200
nCls  = 2
mean1 = array([1., 1.])
sd1   = array([1., 1.])
mean2 = array([3.8, 2.5])
sd2   = array([1., 1.])
data = concatenate( (st.norm.rvs(mean1, sd1, size=(nPts/nCls, 2)), \
                     st.norm.rvs(mean2, sd2, size=(nPts/nCls, 2))), \
                     axis=0 )
target = concatenate( (zeros(nPts/nCls), ones(nPts/nCls)) ).reshape(nPts, 1)
data = concatenate( (data, target), axis=1 )
shuffle(data)
trainData = data[:100,:] 
testData  = data[101:,:]

# Create a Classifier objects
bayMod = mlearn.Ngbayes(trainData[:,:2], trainData[:,2])
logMod = mlearn.BinLogReg(trainData[:,:2], trainData[:,2])

bayMod.train()
logMod.train()

print 
print "Naive (Gaussian) Bayes classifier:"
print bayMod.crossValidate(testData[:,:2], testData[:,2])
print 

print "Binary Logistic Regression classifier:"
print logMod.crossValidate(testData[:,:2], testData[:,2])
print
