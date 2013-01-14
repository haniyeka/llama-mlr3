\name{cptk.cvFolds}
\alias{cptk.cvFolds}
\title{
Cross-validation folds
}
\description{
Take data produced by \code{cptk.input} and amend it with (optionally)
stratified folds for cross-validation.
}
\usage{
cptk.cvFolds(data, nfolds = 10, stratify = T)
}
\arguments{
  \item{data}{
  the data to use. The structure returned by \code{cptk.input}.
}
  \item{nfolds}{
  the number of folds. Defaults to 10.
}
  \item{stratify}{
  whether to stratify the folds. Defaults to TRUE.
}
}
\details{
Partitions the data set into folds. Stratification, if requested, is done by the
best algorithm, i.e. the one with the best performance. The distribution of the
best algorithms in each fold will be approximately the same. The folds are
assembled into training and test sets by combining $n-1$ folds for training and
using the remaining fold for testing. The sets are added to the original data
set and returned.
}
\value{
\item{train}{a list of data sets for training.}
\item{test}{a list of data sets for testing.}
\item{...}{the original members of \code{data}. See \code{\link{cptk.input}}.}
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
trainTest = cptk.cvFolds(satsolvers)

# use 5 folds instead of the default 10
trainTest5 = cptk.cvFolds(satsolvers, 5)

# don't stratify
trainTestU = cptk.cvFolds(satsolvers, stratify=FALSE)
}
\keyword{ ~models }
