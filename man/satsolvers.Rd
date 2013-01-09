\name{satsolvers}
\alias{satsolvers}
\docType{data}
\title{
Example data for the Cork Portfolio Toolkit
}
\description{
Performance data for 19 SAT solvers on 2433 SAT instances.
}
\usage{data(satsolvers)}
\format{
\code{satsolvers} is a list in the format returned by \code{cptk.read} and
expected by the other functions of the Cork Portfolio Toolkit. The list has the
foillowing components.

\describe{
 \item{data:}{The original input data merged. That is, the contents of the files
 read by \code{cptk.read} in a single data frame with the following additonal
 columns.
    \describe{
     \item{best:}{The algorithm with the best performance for each row.}
     \item{*_success:}{For each algorithm whether it was successful on the
     respective row.}
    }}
 \item{features:}{The names of the columns that contain feature values.}
 \item{performance:}{The names of the columns that contain performance data.}
 \item{success:}{The names of the columns indicating whether an algorithm was
 successful.}
}
}
\details{
Performance data for 19 SAT solvers on 2433 SAT instances. For each instance, 36
features were measured. In addition to the performance (time) on each instance,
data on whether a solver timed out on an instance is included.
}
\source{
TODO
%%  ~~ reference to a publication or URL from which the data were obtained ~~
}
\seealso{
\code{\link{cptk.read}}
}
\examples{
data(satsolvers)
}
\keyword{datasets}
