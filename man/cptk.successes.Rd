\name{cptk.successes}
\alias{cptk.successes}
\title{
Success
}
\description{
Was the problem solved successfully using the chosen algorithm?
}
\usage{
cptk.successes(data, predictions)
}
\arguments{
  \item{data}{the data used to induce the model. The same as given to
      \code{cptk.classify}, \code{cptk.cluster} or \code{cptk.regression}.}
  \item{predictions}{the predictions. Can be a list of lists as returned by
      \code{cptk.classify}, \code{cptk.cluster} or \code{cptk.regression} or a
      list as returned by the predictor return value of those functions.}
}
\details{
Returns TRUE if the chosen algorithm successfully solved the problem, FALSE
otherwise for each datum.
}
\value{
A list (or list of lists, depending on the format of \code{predictions}) of the
success values.
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{cptk.misclassificationPenalties}}, \code{\link{cptk.parscores}}
}
\examples{
require(cptk)
require(RWeka)

data(satsolvers)
trainTest = cptk.cvFolds(satsolvers)

cptk.classify(classifier=J48, data=trainTest)
print(sum(unlist(cptk.successes(trainTest, res$predictions))))
}
\keyword{ ~models }
