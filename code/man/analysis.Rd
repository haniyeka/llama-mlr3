\name{analysis}
\alias{contributions}
\title{
Analysis functions
}
\description{
Functions for analysing portfolios.
}
\usage{
contributions(data = NULL, minimize = TRUE, limit = 100)
}
\arguments{
  \item{data}{
  the data to use. The structure returned by \code{input}.
}
\item{minimize}{
  whether the minimum performance value is best. Default true.
}
\item{limit}{
  the maximum number of orderings of the algorithms to evaluate. If the number
  of possible orderings is larger than this, \code{limit} orderings will be
  sampled uniformly from it. Default 100.
}
}
\details{
\code{contributions} analyses the marginal contributions of the algorithms in
the portfolio to its overall performance. For each algorithm, the difference in
performance, in terms of algorithm performance given in the input data, with and
without it for all possible subsets of algorithms is recorded. The marginal
contribution of the algorithm is defined as the mean of those values.

This function requires \eqn{n\cdot 2^{n-1}}{n * 2^(n-1)} orderings to be
evaluated, which becomes very expensive very quickly. The \code{limit} parameter
mitigates that -- if there are too many orderings, a uniformly sampled subset is
evaluated. This is faster, but less accurate.
}
\value{
A table listing the marginal contributions for each algorithm in the portfolio.
The higher the value, the more the respective algorithm contributes to the
overall performance of the portfolio.
}
\author{
Lars Kotthoff
}
\references{
Rahwan, T., Michalak, T. (2013)
A Game Theoretic Approach to Measure Contributions in Algorithm Portfolios.
\emph{Technical Report RR-13-11, University of Oxford}.
}
\examples{
\dontrun{
data(satsolvers)

contributions(satsolvers)
}
}
\keyword{ ~models }
