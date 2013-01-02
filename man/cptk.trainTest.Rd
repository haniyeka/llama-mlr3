\name{cptk.trainTest}
\alias{cptk.trainTest}
\title{
Train / test split
}
\description{
Split a data set into train and test set.
}
\usage{
cptk.trainTest(data, trainpart = 0.6)
}
\arguments{
  \item{data}{
  the data to use. The structure returned by \code{cptk.read}.
}
  \item{trainpart}{
  the fraction of the data to use for training. Default 0.6.
}
}
\details{
Partitions the data set into training and test set according to the specified
fraction. The distribution of the best algorithms in training and test set is
approximately the same, i.e. the sets are stratified.
}
\value{
\item{train }{a (one-element) list of data sets for training.}
\item{test }{a (one-element) list of data sets for testing.}
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
cptk.classify(classifier=J48, data=satsolvers, split=cptk.cvFolds, eval=cptk.trainTest)

# use 50-50 split instead of 60-40
cptk.classify(classifier=J48, data=satsolvers, split=function(x) { return(cptk.trainTest(x, 0.5)) }, eval=cptk.parscore)

## The function is currently defined as
function (data, trainpart = 0.6) 
{
    require(plyr)
    tmp = ddply(data$data, .(best), function(x) {
        n = nrow(x)
        x$train = c(rep.int(1, round(n * trainpart)), rep.int(2, 
            n - round(n * trainpart)))[sample(n, n)]
        x
    })
    parts = split(data$data, tmp$train)
    return(list(train = list(parts[[1]]), test = list(parts[[2]])))
  }
}
\keyword{ ~models }
