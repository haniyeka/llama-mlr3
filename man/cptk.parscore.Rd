\name{cptk.parscore}
\alias{cptk.parscore}
\title{
Penalized average runtime score
}
\description{
Calculates the penalized average runtime score. Commonly used for evaluating
satisfiability solvers on a set of problems.
}
\usage{
cptk.parscore(allperf, chosenperf, chosensuccess, factor = 10)
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
  \item{factor}{
the penalization factor to use for non-successful choices. Default 10.
}
}
\details{
Returns the performance of the chosen algorithm. If the chosen algorithm is not
able to solve the problem successfully, its performance value is multiplied by
the penalization factor. That is, it is assumed that the performance value is
the timeout.
}
\value{
The penalized average runtime.
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.misclassificationPenalty}}, \code{\link{cptk.prediction}},
\code{\link{cptk.success}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
trainTest = cptk.cvFolds(satsolvers)

cptk.classify(classifier=J48, data=trainTest, eval=cptk.parscore)

# use factor of 5 instead of 10.
cptk.classify(classifier=J48, data=trainTest, eval=function(x, y, z) { return(cptk.parscore(x, y, z, 5)) })
}
\keyword{ ~models }
