mlearn
======

Simple machine learning algorithms implemented in Python.

As of 2012-11-24, it has a naive (gaussian) bayes, a binary logistic regression, and 
an adaBoosted stump decision (the stump decision can be used independently, if that's 
what you're into).

Here are some of their characterisitcs and limitations:

* Naive Bayes:   multinomial classification, the targets can be any integer.
* Logit Reg.:    binary classification, the targets must be 0 and 1.
* Boosted stump: binary classification, the targets must be -1 and 1.
