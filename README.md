mlearn
======

Simple machine learning algorithms implemented in Python.

As of 2012-11-24, it has a naive (gaussian) bayes, a binary logistic regression, and 
an adaBoosted stump decision (the stump decision can be used independently, if that's 
what you're into).

Here are some of their characterisitcs and limitations:

                        +-------------+-----------+---------------+
                        | Naive Bayes | Logit Reg | Boosted Stump |
+-----------------------+-------------+-----------+---------------+
|type of classification | mutinomial  | binary    | binary        |
+-----------------------+-------------+-----------+---------------+
|classification tagging | any N num.  | 0 / 1     | -1 / +1       |
+-----------------------+-------------+-----------+---------------+
