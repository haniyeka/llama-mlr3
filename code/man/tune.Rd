\name{tune}
\alias{tuneModel}
\title{
Evaluate the hyperparameters of the machine learning algorithm underlying a model
}
\description{
Functions to evaluate the hyperparameters of the machine learning algorithm
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
\code{tuneModel} evaluates the hyperparameters from the set denoted by
\code{design} of the machine learning algorithm \code{learner} with respect to
the measure \code{metric} for the LLAMA model type \code{llama.fun} on data
\code{ldf}. It uses a nested cross-validation internally; the number of inner
folds is given through \code{nfolds}, the number of outer folds is either
determined by any existing partitions of \code{ldf} or, if none are present, by
\code{nfolds} as well.

During each iteration of the inner and outer cross-validations, all parameter
sets specified in \code{design} are evaluated The mean performance over all
instances in the data is logged and returned. The evaluation across the folds
sets will be parallelized automatically if a suitable backend for parallel
computation is loaded. The \code{parallelMap} level is "llama.tune".

The performance value with respect to the metric is normalised by the
performance with respect to the metric of the single best solver. This is done
to facilitate comparison of the numbers across folds.

The return value is a data frame that contains all the information gathered
during the evaluation run. For each parameter set and outer fold, the score on
the inner cross-validation and the outer train/test split (denoted by the
particular fold of the outer cross-validation) is reported. This allows to
compare the performance of the parameter sets and the generalisation
performance. That is, if the (normalised) scores for the inner cross-validation
and the outer train/test split are significantly different, it may denote an
unstable parameter set.

The choice of the \emph{best} parameter set is left up to the user, based on the
information provided.
}
\value{
A data frame with the parameter values, the fold, and the performance achieved
in the outer and inner cross-validations.
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
