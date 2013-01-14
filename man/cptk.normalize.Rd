\name{cptk.normalize}
\alias{cptk.normalize}
\title{
Normalize features
}
\description{
Normalize input data so that the values for all features cover the same range -1
to 1.
}
\usage{
cptk.normalize(rawfeatures, meta = NULL)
}
\arguments{
  \item{rawfeatures}{
data frame with the feature values to normalize.
}
  \item{meta}{
meta data to use for the normalization. If supplied should be a list with
members \code{minValues} that contains the minimum values for all features and
\code{maxValues} that contains the maximum values for all features. Will be
computed if not supplied.
}
}
\details{
\code{normalize} subtracts the minimum (supplied or computed) from all values of
a feature, divides by the difference between maximum and minimum, multiplies by
2 and subtracts 1. The range of the values for all features will be -1 to 1.
}
\value{
\item{features}{the normalized feature vectors.}
\item{meta}{the minimum and maximum values for each feature before
normalization. Can be used in subsequent calls to \code{normalize} for new
data.}
}
\author{
Lars Kotthoff
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
trainTest = cptk.cvFolds(satsolvers)

cptk.cluster(clusterer=XMeans, data=trainTest, pre=cptk.normalize, eval=cptk.parscore)
}
\keyword{ ~models }
