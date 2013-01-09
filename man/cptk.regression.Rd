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
require(RWeka)

data(satsolvers)
res = cptk.regression(regressor=make_Weka_classifier("weka/classifiers/functions/LinearRegression"), data=satsolvers, split=cptk.cvFolds, eval=list(cptk.successes, cptk.misclassificationPenalty, cptk.parscore))
print(res$eval)
print(res$predictor(subset(satsolvers$data, TRUE, satsolvers$features)))

## The function is currently defined as
function (regressor = NULL, data = NULL, split = NULL, pre = NULL, 
    eval = NULL, combine = min) 
{
    splitdata = split(data)
    if (class(eval) != "list") {
        eval = list(eval)
    }
    scores = rep.int(0, length(eval))
    for (i in 1:length(splitdata$train)) {
        performancePredictions = matrix(nrow = nrow(splitdata$test[[i]]), 
            ncol = length(data$performance))
        trf = list(features = subset(splitdata$train[[i]], T, 
            data$features))
        if (!is.null(pre)) {
            trf = pre(trf$features)
        }
        tsf = list(features = subset(splitdata$test[[i]], T, 
            data$features))
        if (!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        for (id in 1:length(data$performance)) {
            model = regressor(splitdata$train[[i]][[data$performance[id]]] ~ 
                ., trf$features)
            performancePredictions[, id] = predict(model, tsf$features)
        }
        predictions = apply(performancePredictions, 1, function(x) {
            head(which(x \%in\% combine(x)), 1)
        })
        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id, ], T, data$performance)
            successes = subset(splitdata$test[[i]][id, ], T, 
                data$success)
            for (j in 1:length(eval)) {
                scores[j] = scores[j] + as.numeric(eval[[j]](perfs, 
                  perfs[predictions[id]], successes[predictions[id]]))
            }
        }
    }
    fs = list(features = subset(data$data, T, data$features))
    if (!is.null(pre)) {
        fs = pre(fs$features)
    }
    models = lapply(1:length(data$performance), function(id) {
        return(regressor(data$data[[data$performance[id]]] ~ 
            ., fs$features))
    })
    return(list(eval = scores, predictor = function(x) {
        tsf = list(features = subset(x, T, data$features))
        if (!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }
        performancePredictions = matrix(nrow = nrow(x), ncol = length(data$performance))
        for (id in 1:length(data$performance)) {
            performancePredictions[, id] = predict(models[[id]], 
                tsf$features)
        }
        return(apply(performancePredictions, 1, function(y) {
            factor(data$performance[head(which(y \%in\% combine(y)), 
                1)])
        }))
    }))
  }
}
\keyword{ ~regression }
