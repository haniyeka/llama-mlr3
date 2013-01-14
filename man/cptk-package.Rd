\name{cptk-package}
\alias{cptk-package}
\alias{cptk}
\docType{package}
\title{
Cork Portfolio Toolkit
}
\description{
The Cork Portfolio Toolkit provides functionality to read and process
performance data for algorithms, facilitate building models that predict which
algorithm to use in which scenario and ways of evaluating them.
}
\details{
\tabular{ll}{
Package: \tab cptk\cr
Type: \tab Package\cr
Version: \tab 0.1\cr
Date: \tab 2013-01-01\cr
Depends: \tab plyr\cr
Suggests: \tab RWeka\cr
License: \tab BSD\cr
}
The package provides functions to read performance data, build performance
models that enable selection of algorithms (using external machine learning
functions) and evaluate those models.

Data is input using \code{cptk.input} and can then be used to learn
performance models. There are currently three main ways to create models.
Classification (\code{cptk.classify}) creates a single machine learning model
that predicts the algorithm to use as a label. Clustering (\code{cptk.cluster})
clusters the problems to solve and assigns the best algorithm to each cluster.
Regression (\code{cptk.regression}) trains separate models for all available
algorithms, predicts the performance on a problem independently and chooses the
algorithm with the best predicted performance.

Various functions to split the data into training and test set(s) and to
evaluate the performance of the learned models are provided.

Please note that this is an alpha release. Bugs should be expected and the code
used with care. More sophisticated functionality has not been implemented yet.
Function names and interfaces may change in future versions.

At the moment, the implementation of the functions is very much geared towards
\code{RWeka}, an R package to interface with the Weka machine learning toolkit.
While in theory using other packages and implementations should be possible,
this may not be so in practice.
}
\author{
Lars Kotthoff

Maintainer: Lars Kotthoff <larsko@4c.ucc.ie>
}
\references{
Lars Kotthoff, Algorithm Selection for Combinatorial Search Problems: A survey,
2012
}
\keyword{ package }
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
trainTest = cptk.cvFolds(satsolvers)

res = cptk.classify(classifier=J48, data=trainTest)
# print the total number of successes
print(sum(unlist(cptk.successes(trainTest, res$predictions))))
# print the total misclassification penalty
print(sum(unlist(cptk.misclassificationPenalties(trainTest, res$predictions))))
# print the total PAR10 score
print(sum(unlist(cptk.parscores(trainTest, res$predictions))))

# print predictions on the entire data set
print(res$predictor(subset(satsolvers$data, TRUE, satsolvers$features)))
}
