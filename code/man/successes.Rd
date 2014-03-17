\name{successes}
\alias{successes}
\title{
Success
}
\description{
Was the problem solved successfully using the chosen algorithm?
}
\usage{
successes(data, model)
}
\arguments{
  \item{data}{the data used to induce the model. The same as given to
      \code{classify}, \code{cluster} or \code{regression}.}
  \item{model}{the algorithm selection model. Can be either a model
      returned by one of the model-building functions or a function that returns
      predictions such as \code{vbs} or the predictor function of a trained
      model.}
}
\details{
Returns TRUE if the chosen algorithm successfully solved the problem instance,
FALSE otherwise for each problem instance.
}
\value{
A list of the success values.
}
\author{
Lars Kotthoff
}
\seealso{
\code{\link{misclassificationPenalties}}, \code{\link{parscores}}
}
\examples{
\dontrun{
library(RWeka)

data(satsolvers)
trainTest = cvFolds(satsolvers)

model = classify(classifier=J48, data=trainTest)
sum(successes(trainTest, model))
}
}
\keyword{ ~models }
