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
cptk.cluster(clusterer = NULL, data = NULL, pre = NULL)
}
\arguments{
  \item{clusterer}{
  the clustering function to use. Must accept a data frame with features. Return
  value should be a structure that can be given to \code{predict} along with new
  data. See examples.
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
\code{cptk.cluster} takes \code{data} and processes it using \code{pre} (if
supplied). \code{clusterer} is called to cluster the data. For each cluster, the
best algorithm is identified as the one with the best overall performance on the
problems in that cluster. The learned model is used to cluster the test data and
predict algorithms accordingly.
}
\value{
 \item{predictions}{a list of lists with the predictions for each test set.}
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
trainTest = cptk.cvFolds(satsolvers)

res = cptk.cluster(clusterer=XMeans, data=trainTest, pre=cptk.normalize)
# the total number of successes
sum(unlist(cptk.successes(trainTest, res$predictions)))
# predictions on the entire data set
res$predictor(subset(satsolvers$data, TRUE, satsolvers$features))

require(flexclust)
res = cptk.cluster(clusterer=function(x) { kcca(x, length(satsolvers$performance)) }, data=trainTest, pre=cptk.normalize)
}
\keyword{ ~cluster }
