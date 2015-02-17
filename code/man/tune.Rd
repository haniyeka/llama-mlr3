\name{tune}
\alias{tuneModel}
\title{
Tune the hyperparameters of the machine learning algorithm underlying a model
}
\description{
Functions to tune the hyperparameters of the machine learning algorithm
underlying a model with respect to a performance measure.
}
\usage{
tuneModel(ldf, llama.fun, learner, design, metric = parscores, nfolds = 10L,
    quiet = FALSE)
}
\arguments{
  \item{ldf}{the LLAMA data to use. The structure returned by \code{input}.}
  \item{llama.fun}{the LLAMA model building function.}
  \item{learner}{the mlr learner to use.}
  \item{design}{the data frame denoting the parameter values to try. Can be
    produced with the \code{ParamHelpers} package. See examples.}
  \item{metric}{the metric used to evaluate the model. Can be one of
    \code{misclassificationPenalties}, \code{parscores} or \code{successes}.}
  \item{nfolds}{the number of folds. Defaults to 10. If -1 is given,
    leave-one-out cross-validation folds are produced.}
  \item{quiet}{whether to output information on the intermediate values and
    progress during tuning.}
}
\details{
\code{tuneModel} finds the hyperparameters from the set denoted by \code{design}
of the machine learning algorithm \code{learner} that give the best performance
with respect to the measure \code{metric} for the LLAMA model type
\code{llama.fun} on data \code{ldf}. It uses a nested cross-validation
internally; the number of inner folds is given through \code{nfolds}, the number
of outer folds is either determined by any existing partitions of \code{ldf} or,
if none are present, by \code{nfolds} as well.

During each iteration of the inner cross-validation, all parameter sets
specified in \code{design} are evaluated and the one with the best performance
value chosen. The mean performance over all instances in the data is logged for
all evaluations. This parameter set is then used to build and evaluate a model
in the outer cross-validation.

The best parameter set, corresponding mean performance, and LLAMA model for each
fold of the outer cross-validation are returned.
}
\value{
A list with one element for each outer fold. Each item has members
\code{parameters}, which contains the parameter set that achieved the best
performance, \code{score}, which contains the corresponding performance, and
\code{model}, which contains the LLAMA model trained during this fold.
}
\author{
Lars Kotthoff
}
\examples{
if(Sys.getenv("RUN_EXPENSIVE") == "true") {
library(ParamHelpers)
data(satsolvers)

learner = makeLearner("classif.J48")
# parameter set for J48
ps = makeParamSet(makeIntegerParam("M", lower = 1, upper = 100))
# generate 10 random parameter sets
design = generateRandomDesign(10, ps)
# tune with respect to PAR10 score (default) with 10 outer and inner folds
# (default)
res = tuneModel(satsolvers, classify, learner, design)
}
}
\keyword{ models }
