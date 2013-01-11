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
  \code{cptk.trainTest} or \code{cptk.cvFolds}. Alternatively, a data set that
  has been split already by one of these functions.
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
res = cptk.cluster(clusterer=XMeans, data=satsolvers, split=cptk.cvFolds, pre=cptk.normalize, eval=list(cptk.successes, cptk.misclassificationPenalty, cptk.parscore))
print(res$eval)
print(res$predictor(subset(satsolvers$data, TRUE, satsolvers$features)))

require(flexclust)
res = cptk.cluster(clusterer=function(x) { kcca(x, length(satsolvers$performance)) }, data=satsolvers, split=cptk.cvFolds, pre=cptk.normalize, eval=list(cptk.successes, cptk.misclassificationPenalty, cptk.parscore))
}
\keyword{ ~cluster }
