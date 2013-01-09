\name{cptk.cluster}
\alias{cptk.cluster}
\title{
Cluster model
}
\description{
Build a cluster model that predicts the algorithm to use based on the features
of the problem.
}
\usage{
cptk.cluster(clusterer = NULL, data = NULL, split = NULL, pre = NULL, eval = NULL)
}
\arguments{
  \item{clusterer}{
  the clustering function to use. Must accept a data frame with features. Return
  value should be a structure that can be given to \code{predict} along with new
  data. See examples.
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
\code{cptk.cluster} takes \code{data} and processes it using \code{pre} (if
supplied). The preprocessed data is then split into training and test sets using
\code{split} and \code{clusterer} is called to cluster the data. For each
cluster, the best algorithm is identified as the one with the best overall
performance on the problems in that cluster. The learned model is evaluated on
each of \code{eval}.
}
\value{
 \item{eval}{a list with the values for the evaluation functions in the same
 order as in the \code{eval} argument. If evaluated on more than one test set,
 the sum across all test sets.}
 \item{predictor}{a function that encapsulates the model learned on the
 \emph{entire} data set. Can be called with data for the same features with the
 same feature names as the training data to obtain predictions.}
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.classify}}, \code{\link{cptk.regression}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
res = cptk.cluster(clusterer=make_Weka_clusterer("weka/clusterers/EM"), data=satsolvers, split=cptk.cvFolds, pre=cptk.normalize, eval=cptk.parscore)
print(res$eval)
print(res$predictor(subset(satsolvers$data, TRUE, satsolvers$features)))

## The function is currently defined as
function (clusterer = NULL, data = NULL, split = NULL, pre = NULL, 
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
        model = clusterer(trf$features)
        trainclusters = predict(model)
        best = by(splitdata$train[[i]], trainclusters, function(x) {
            names(sort(table(x$best), decreasing = T))[1]
        })
        tsf = list(features = subset(splitdata$test[[i]], T, 
            data$features))
        if (!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        testclusters = predict(model, tsf$features)
        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id, ], T, data$performance)
            successes = subset(splitdata$test[[i]][id, ], T, 
                data$success)
            for (j in 1:length(eval)) {
                scores[j] = scores[j] + as.numeric(eval[[j]](perfs, 
                  perfs[which(data$performance == best[[which(names(best) == 
                    testclusters[id])]])], successes[which(data$performance == 
                    best[[which(names(best) == testclusters[id])]])]))
            }
        }
    }
    fs = list(features = subset(data$data, T, data$features))
    if (!is.null(pre)) {
        fs = pre(fs$features)
    }
    model = clusterer(fs$features)
    clusters = predict(model)
    best = by(data$data, clusters, function(x) {
        names(sort(table(x$best), decreasing = T))[1]
    })
    return(list(eval = scores, predictor = function(x) {
        tsf = list(features = subset(x, T, data$features))
        if (!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }
        return(sapply(predict(model, tsf$features), function(y) {
            return(factor(best[[which(names(best) == y)]]))
        }))
    }))
  }
}
\keyword{ ~cluster }
