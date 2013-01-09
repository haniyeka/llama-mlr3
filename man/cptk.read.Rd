\name{cptk.read}
\alias{cptk.read}
\title{
Read data
}
\description{
Reads performance data that can be used to train and evaluate models.
}
\usage{
cptk.read(featurefilename, perffilename, successfilename = NULL, minimize = T)
}
\arguments{
  \item{featurefilename}{
  comma-separated file that contains the feature values for each problem.
}
  \item{perffilename}{
  comma-separated file that contains the performance values for each problem.
}
  \item{successfilename}{
  comma-separated file that contains the success values (true/false) for each
  problem. Optional.
}
  \item{minimize}{
  whether the minimum performance value is best. Default true.
}
}
\details{
\code{cptk.read} takes a list of file names and reads the data. The files are
expected to contain comma-separated values with headers (readable through
\code{read.csv}). The feature and performance data are joined by looking for
common column names in the two files (usually an ID of the problem). For each
problem, the best algorithm according to the read performance data is computed.

The file that describes whether an algorithm was successful on a problem is
optional. If \code{cptk.parscore} or \code{cptk.successes} are to be used to
evaluate the learned models, this argument is required however and will lead to
error messages if not supplied.
}
\value{
\item{data}{the combined data (features, performance, successes).}
\item{features}{a list of names denoting problem features.}
\item{performance}{a list of names denoting algorithm performances.}
\item{success}{a list of names denoting algorithm successes.}
}
\author{
Lars Kotthoff
}
\examples{
require(cptk)
# features.csv looks something like
# ID,width,height
# 0,1.2,3
# ...
# performance.csv
# ID,alg1,alg2
# 0,2,5
# ...
# success.csv
# ID,alg1,alg2
# 0,T,F
# ...
#cptk.read("features.csv", "performance.csv", "success.csv")
}
\keyword{ ~models }
