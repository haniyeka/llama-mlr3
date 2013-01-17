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
cptk.classify(classifier = NULL, data = NULL, pre = function(x, y=NULL) { list(features=x) })
}
\arguments{
  \item{classifier}{
  the classifier function to use. Must accept a formula of the values to predict
  and a data frame with features. Return value should be a structure that can be
  given to \code{predict} along with new data. See examples.

  The argument can also be a list of such classifiers.
}
  \item{data}{
  the data to use with training and test sets. The structure returned by
  \code{cptk.trainTest} or \code{cptk.cvFolds}.
}
  \item{pre}{
  a function to preprocess the data. Currently only \code{cptk.normalize}.
  Optional. Does nothing by default.
}
}
\details{
\code{cptk.classify} takes the training and test sets in \code{data} and
processes it using \code{pre} (if supplied). \code{classifier} is called to
induce a classifier. The learned model is used to make predictions on the test
set(s).

If a list of classifiers is supplied in \code{classifier}, ensemble
classification is performed. That is, the models are trained and used to make
predictions independently. For each instance, the final prediction is determined
by majority vote of the predictions of the individual models -- the class that
occurs most often is chosen. If the list given as \code{classifier} contains a
member \code{.combine} that is a function, it is assumed to be a classifier with
the same properties as the other ones and will be used to combine the ensemble
predictions instead of majority voting.
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
\references{
Kotthoff, L., Miguel, I., Nightingale, P. (2010)
Ensemble Classification for Constraint Solver Configuration.
\emph{16th International Conference on Principles and Practices of Constraint Programming}, 321--329.
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

# ensemble classification
rese = cptk.classify(classifier=list(J48, IBk, svm), data=trainTest)

# ensemble classification with a classifier to combine predictions
rese = cptk.classify(classifier=list(J48, IBk, svm, .combine=J48), data=trainTest)
}
\keyword{ ~models }
