\name{cptk.cvFolds}
\alias{cptk.cvFolds}
\title{
Cross-validation folds
}
\description{
Take data produced by \code{cptk.read} and create stratified folds for
cross-validation.
}
\usage{
cptk.cvFolds(data, nfolds = 10)
}
\arguments{
  \item{data}{
  the data to use. The structure returned by \code{cptk.read}.
}
  \item{nfolds}{
  the number of folds. Defaults to 10.
}
}
\details{
Partitions the data set into folds. Stratification is done by the best
algorithm, i.e. the one with the best performance. The distribution of the best
algorithms in each fold will be approximately the same. The folds are assembled
into training and test sets by combining $n-1$ folds for training and using the
remaining fold for testing.
}
\value{
\item{train}{a list of data sets for training.}
\item{test}{a list of data sets for testing.}
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.trainTest}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
cptk.classify(classifier=J48, data=satsolvers, split=cptk.cvFolds, eval=cptk.parscore)

# use 5 folds instead of the default 10
cptk.classify(classifier=J48, data=satsolvers, split=function(x) { return(cptk.cvFolds(x, 5)) }, eval=cptk.parscore)
}
\keyword{ ~models }
