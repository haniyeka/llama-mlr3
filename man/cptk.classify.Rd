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
cptk.classify(classifier = NULL, data = NULL, split = NULL, pre = NULL, eval = NULL)
}
\arguments{
  \item{classifier}{
  the classifier function to use. Must accept a formula of the values to predict
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
}
\details{
\code{cptk.classify} takes \code{data} and processes it using \code{pre} (if
supplied). The preprocessed data is then split into training and test sets using
\code{split} and \code{classifier} is called to induce a classifier. The learned
model is evaluated on each of \code{eval}.
}
\value{
 \item{eval}{a list with the values for the evaluation functions in the same
 order as in the \code{eval} argument. If evaluated on more than one test set,
 the sum across all test sets.}
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
res = cptk.classify(classifier=J48, data=satsolvers, split=cptk.cvFolds, eval=list(cptk.successes, cptk.misclassificationPenalty, cptk.parscore))
print(res$eval)
print(res$predictor(subset(satsolvers$data, TRUE, satsolvers$features)))

## The function is currently defined as
function (classifier = NULL, data = NULL, split = NULL, pre = NULL, 
    eval = NULL) 
{
    splitdata = split(data)
    if (class(eval) != "list") {
        eval = list(eval)
    }
    scores = rep.int(0, length(eval))
    for (i in 1:length(splitdata$train)) {
        trf = list(features = subset(splitdata$train[[i]], T, 
            data$features))
        if (!is.null(pre)) {
            trf = pre(trf$features)
        }
        model = classifier(splitdata$train[[i]]$best ~ ., trf$features)
        tsf = list(features = subset(splitdata$test[[i]], T, 
            data$features))
        if (!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        predictions = predict(model, tsf$features)
        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id, ], T, data$performance)
            successes = subset(splitdata$test[[i]][id, ], T, 
                data$success)
            for (j in 1:length(eval)) {
                scores[j] = scores[j] + as.numeric(eval[[j]](perfs, 
                  perfs[which(data$performance == predictions[id])], 
                  successes[which(data$performance == predictions[id])]))
            }
        }
    }
    fs = list(features = subset(data$data, T, data$features))
    if (!is.null(pre)) {
        fs = pre(fs$features)
    }
    model = classifier(data$data$best ~ ., fs$features)
    return(list(eval = scores, predictor = function(x) {
        tsf = list(features = subset(x, T, data$features))
        if (!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }
        return(predict(model, tsf$features))
    }))
  }
}
\keyword{ ~models }
