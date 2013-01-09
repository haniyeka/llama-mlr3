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
cptk.regression(regressor = NULL, data = NULL, split = NULL, pre = NULL, eval = NULL, combine = min)
}
\arguments{
  \item{regressor}{
  the regression function to use. Must accept a formula of the values to predict
  and a data frame with features. Return value should be a structure that can be
  given to \code{predict} along with new data. See examples.
}
  \item{data}{
  the data to use. The structure returned by \code{cptk.read}.
}
  \item{split}{
  a function to split the data into training and test set(s). One of
  \code{cptk.trainTest} or \code{cptk.cvFolds}.
}
  \item{pre}{
  a function to preprocess the data. Currently only \code{cptk.normalize}.
  Optional.
}
  \item{eval}{
  a list of functions to use to evaluate the learned model. Currently supported
  are \code{cptk.misclassificationPenalty}, \code{cptk.parscore} and
  \code{cptk.successes}.
}
  \item{combine}{
  the function used to combine the predictions of the individual regression
  models. Default \code{min}. See details.
}
}
\details{
\code{cptk.regression} takes \code{data} and processes it using \code{pre} (if
supplied). The preprocessed data is then split into training and test sets using
\code{split} and \code{regressor} is called to induce separate regression models
for each of the algorithms to predict its performance. The predicted
performances are combined using \code{combine} to choose the best algorithm.
That is, the algorithm with the predicted performance that \code{combine}
chooses is selected. The learned model is evaluated on each of \code{eval}.
}
\value{
 \item{eval}{a list with the values for the evaluation functions in the same
 order as in the \code{eval} argument. If evaluated on more than one test set,
 the sum across all test sets.}
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
res = cptk.regression(regressor=lm, data=satsolvers, split=cptk.cvFolds, eval=list(cptk.successes, cptk.misclassificationPenalty, cptk.parscore))
print(res$eval)
print(res$predictor(subset(satsolvers$data, TRUE, satsolvers$features)))

require(RWeka)
res = cptk.regression(regressor=make_Weka_classifier("weka/classifiers/functions/LinearRegression"), data=satsolvers, split=cptk.cvFolds, eval=list(cptk.successes, cptk.misclassificationPenalty, cptk.parscore))
}
\keyword{ ~regression }
