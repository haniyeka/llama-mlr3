\name{cptk.regression}
\alias{cptk.regression}
\title{
Regression model
}
\description{
Build a regression model that predicts the algorithm to use based on the
features of the problem.
}
\usage{
cptk.regression(regressor = NULL, data = NULL, pre = NULL, combine = min)
}
\arguments{
  \item{regressor}{
  the regression function to use. Must accept a formula of the values to predict
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
  \item{combine}{
  the function used to combine the predictions of the individual regression
  models. Default \code{min}. See details.
}
}
\details{
\code{cptk.regression} takes \code{data} and processes it using \code{pre} (if
supplied). \code{regressor} is called to induce separate regression models for
each of the algorithms to predict its performance. The predicted performances
are combined using \code{combine} to choose the best algorithm. That is, the
algorithm with the predicted performance that \code{combine} chooses is
selected.
}
\value{
 \item{predictions}{a list of lists with the predictions for each test set.}
 \item{predictor}{a function that encapsulates the regression model learned on
 the \emph{entire} data set. Can be called with data for the same features with
 the same feature names as the training data to obtain predictions.}
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.classify}}, \code{\link{cptk.cluster}}
}
\examples{
require(cptk)
data(satsolvers)
trainTest = cptk.cvFolds(satsolvers)

res = cptk.regression(regressor=lm, data=trainTest)
# the total number of successes
sum(unlist(cptk.successes(trainTest, res$predictions)))
# predictions on the entire data set
res$predictor(subset(satsolvers$data, TRUE, satsolvers$features))

require(RWeka)
res = cptk.regression(regressor=LinearRegression, data=trainTest)
}
\keyword{ ~regression }
