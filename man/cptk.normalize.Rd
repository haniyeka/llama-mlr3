\name{cptk.normalize}
\alias{cptk.normalize}
\title{
Normalize
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
cptk.cluster(clusterer=make_Weka_clusterer("weka/clusterers/EM"), data=satsolvers, split=cptk.cvFolds, pre=cptk.normalize, eval=cptk.parscore)

## The function is currently defined as
function (rawfeatures, meta = NULL) 
{
    if (is.null(meta) || is.null(meta$minValues)) {
        meta$minValues = apply(rawfeatures, 2, min, na.rm = T)
    }
    if (is.null(meta) || is.null(meta$maxValues)) {
        meta$maxValues = apply(rawfeatures, 2, max, na.rm = T)
    }
    tmp = rawfeatures
    for (i in names(rawfeatures)) {
        tmp[[i]] = if ((meta$maxValues[[i]] - meta$minValues[[i]]) == 
            0) {
            rep.int(0, length(rawfeatures[[i]]))
        }
        else {
            ((rawfeatures[[i]] - meta$minValues[[i]])/(meta$maxValues[[i]] - 
                meta$minValues[[i]]) * 2) - 1
        }
    }
    return(list(features = tmp, meta = meta))
  }
}
\keyword{ ~models }
