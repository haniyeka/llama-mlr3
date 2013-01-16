\name{cptk.featureFilter}
\alias{cptk.featureFilter}
\title{
Feature filtering
}
\description{
Take data produced by \code{cptk.input} and determine the most predictive
features (with respect to the best algorithm).
}
\usage{
cptk.featureFilter(filter, data)
}
\arguments{
  \item{filter}{
  the filtering algorithm to use. Must accept a formula of the values to predict
  and a data frame with features. Return value should be the list of selected
  features.
}
  \item{data}{
  the data to use. The structure returned by \code{cptk.input}.
}
}
\details{
Runs the specified feature filtering algorithm on the data set with all
features. The subset of the features chosen by the filter are saved in the
\code{features} member, the old features (i.e. the full feature set) in
\code{original_features}. The data structure modified like this is returned.
}
\value{
\item{features}{list of the most predictive features.}
\item{original_features}{the list of the orignal features.}
\item{...}{the original members of \code{data}. See \code{\link{cptk.input}}.}
}
\author{
Lars Kotthoff
}
\examples{
require(cptk)
require(FSelector)

data(satsolvers)
d = cptk.featureFilter(filter=cfs, satsolvers)
d$features
}
\keyword{ ~models }
