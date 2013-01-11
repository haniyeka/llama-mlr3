\name{cptk.success}
\alias{cptk.success}
\title{
Success
}
\description{
Was the problem solved successfully using the chosen algorithm?
}
\usage{
cptk.success(allperf, chosenperf, chosensuccess)
}
\arguments{
  \item{allperf}{
a list of performances of all algorithms on the problem.
}
  \item{chosenperf}{
the performance of the chosen algorithm on the problem.
}
  \item{chosensuccess}{
whether the chosen algorithm successfully solved the problem.
}
}
\details{
Returns 1 if the chosen algorithm successfully solved the problem, 0 otherwise.
}
\value{
1 if the chosen algorithm successfully solved the problem, 0 otherwise.
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.misclassificationPenalty}}, \code{\link{cptk.parscore}},
\code{\link{cptk.prediction}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
cptk.classify(classifier=J48, data=satsolvers, split=cptk.cvFolds, eval=cptk.success)
}
\keyword{ ~models }
