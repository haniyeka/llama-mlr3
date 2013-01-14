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
cptk.classify(classifier = NULL, data = NULL, pre = NULL)
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
}
\details{
\code{cptk.classify} takes the training and test sets in \code{data} and
processes it using \code{pre} (if supplied). \code{classifier} is called to
induce a classifier. The learned model is used to make predictions on the test
set(s).
}
\value{
 \item{predictions}{a list of lists with the predictions for each test set.}
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

res = cptk.classify(classifier=J48, data=trainTest)
# the total number of successes
sum(unlist(cptk.successes(trainTest, res$predictions)))
# predictions on the entire data set
res$predictor(subset(satsolvers$data, TRUE, satsolvers$features))

require(e1071)
res = cptk.classify(classifier=svm, data=trainTest)
}
\keyword{ ~models }
