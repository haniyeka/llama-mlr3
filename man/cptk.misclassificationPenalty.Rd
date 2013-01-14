\name{cptk.misclassificationPenalty}
\alias{cptk.misclassificationPenalty}
\title{
Misclassification penalty
}
\description{
Calculates the penalty incurred because of making incorrect decisions, i.e.
choosing suboptimal algorithms.
}
\usage{
cptk.misclassificationPenalty(allperf, chosenperf, chosensuccess, minimize = T)
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
  \item{minimize}{
is the minimum performance value the best or worst? Default T for best.
}
}
\details{
Compares the performance of the chosen algorithm to the performance of the best
algorithm. Returns the absolute difference. This denotes the penalty for
choosing a suboptimal algorithm, e.g. the additional time required to solve a
problem or reduction in solution quality incurred.
}
\value{
The misclassification penalty.
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.parscore}}, \code{\link{cptk.prediction}},
\code{\link{cptk.success}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
trainTest = cptk.cvFolds(satsolvers)

cptk.classify(classifier=J48, data=trainTest, eval=cptk.misclassificationPenalty)

# maximum performance value is best
cptk.classify(classifier=J48, data=trainTest, eval=function(x, y, z) { return(cptk.misclassificationPenalty(x, y, z, FALSE)) })
}
\keyword{ ~models }
