\name{cptk.trainTest}
\alias{cptk.trainTest}
\title{
Train / test split
}
\description{
Split a data set into train and test set.
}
\usage{
cptk.trainTest(data, trainpart = 0.6, stratify = T)
}
\arguments{
  \item{data}{
  the data to use. The structure returned by \code{cptk.input}.
}
  \item{trainpart}{
  the fraction of the data to use for training. Default 0.6.
}
  \item{stratify}{
  whether to stratify the folds. Defaults to TRUE.
}
}
\details{
Partitions the data set into training and test set according to the specified
fraction. The training and test set are added to the original data and returned.
If requested, the distribution of the best algorithms in training and test set
is approximately the same, i.e. the sets are stratified.
}
\value{
\item{train}{a (one-element) list of data sets for training.}
\item{test}{a (one-element) list of data sets for testing.}
\item{...}{the original members of \code{data}. See \code{\link{cptk.input}}.}
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.cvFolds}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
trainTest = cptk.trainTest(satsolvers)

# use 50-50 split instead of 60-40
trainTest1 = cptk.trainTest(satsolvers, 0.5)

# don't stratify
trainTestU = cptk.trainTest(satsolvers, stratify=FALSE)
}
\keyword{ ~models }
