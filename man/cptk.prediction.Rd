\name{cptk.prediction}
\alias{cptk.prediction}
\title{
Prediction
}
\description{
Returns the performance of the predicted algorithm for a problem.
}
\usage{
cptk.prediction(allperf, chosenperf, chosensuccess)
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
Returns the performance \code{chosenperf} of the algorithm that was chosen.
}
\value{
The performance of the predicted algorithm.
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.misclassificationPenalty}}, \code{\link{cptk.parscore}},
\code{\link{cptk.success}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
cptk.classify(classifier=J48, data=satsolvers, split=cptk.cvFolds, eval=cptk.prediction)
}
\keyword{ ~models }
