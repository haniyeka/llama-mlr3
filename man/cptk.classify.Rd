\name{cptk.classify}
\alias{cptk.classify}
\title{
Classification model
}
\description{
Build a classification model that predicts the algorithm to use based on the
features of the problem.
}
\usage{
cptk.classify(classifier = NULL, data = NULL, pre = NULL, eval = NULL)
}
\arguments{
  \item{classifier}{
  the classifier function to use. Must accept a formula of the values to predict
  and a data frame with features. Return value should be a structure that can be
  given to \code{predict} along with new data. See examples.
}
  \item{data}{
  the data to use with training and test sets. The structure returned by
  \code{cptk.trainTest} or \code{cptk.cvFolds}.
}
  \item{pre}{
  a function to preprocess the data. Currently only \code{cptk.normalize}.
  Optional.
}
  \item{eval}{
  a list of functions to use to evaluate the learned model. Currently supported
  are \code{cptk.misclassificationPenalty}, \code{cptk.parscore},
  \code{cptk.prediction} and \code{cptk.success}.
}
}
\details{
\code{cptk.classify} takes the training and test sets in \code{data} and
processes it using \code{pre} (if supplied). \code{classifier} is called to
induce a classifier. The learned model is evaluated on each of \code{eval}.
}
\value{
 \item{eval}{a list with a list of values for the evaluation functions in the
 same order as in the \code{eval} argument. That is, there is one element for
 each function in \code{eval} and each of those elements is a list with the
 results of the evaluation for each problem it was evaluated on.}
 \item{predictor}{a function that encapsulates the classifier learned on the
 \emph{entire} data set. Can be called with data for the same features with the
 same feature names as the training data to obtain predictions.}
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.cluster}}, \code{\link{cptk.regression}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
trainTest = cptk.cvFolds(satsolvers)

res = cptk.classify(classifier=J48, data=trainTest, eval=list(cptk.success, cptk.misclassificationPenalty, cptk.parscore))
# print the total number of successes
print(sum(res$eval[[1]]))
print(res$predictor(subset(satsolvers$data, TRUE, satsolvers$features)))

require(e1071)
res = cptk.classify(classifier=svm, data=trainTest, eval=list(cptk.success, cptk.misclassificationPenalty, cptk.parscore))
}
\keyword{ ~models }
