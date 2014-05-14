makeRLearner.classif.test = function() {
    makeRLearnerClassif(cl = "classif.test", package="llama", par.set=makeParamSet(), numerics=T, factors=T)
}
trainLearner.classif.test = function(.learner, .task, .subset, .weights, ...) { }
predictLearner.classif.test = function(learner, model, newdata) {
    return(rep.int(factor(model$factor.levels$target[1]), nrow(newdata)))
}
testclassifier = makeLearner("classif.test")

makeRLearner.classif.otest = function() {
    makeRLearnerClassif(cl = "classif.otest", package="llama", par.set=makeParamSet(), numerics=T, factors=T)
}
trainLearner.classif.otest = function(.learner, .task, .subset, .weights, ...) { }
predictLearner.classif.otest = function(learner, model, newdata) {
    return(factor(rep.int("bar", nrow(newdata))))
}
othertestclassifier = makeLearner("classif.otest")

test_that("classifyPairs classifies", {
    fold = data.frame(a=rep.int(0, 10), best=rep.int("foo", 10), foo=rep.int(0, 10), bar=rep.int(1, 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    res = classifyPairs(classifier=testclassifier, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="foo", score=1) }) })))
})

test_that("classifyPairs returns predictor", {
    fold = data.frame(a=rep.int(0, 10), best=rep.int("foo", 10), foo=rep.int(0, 10), bar=rep.int(1, 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    res = classifyPairs(classifier=testclassifier, d)
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="foo", score=1) })))
})

test_that("classifyPairs raises error without classifier", {
    expect_error(classifyPairs(), "No classifier given!")
})

test_that("classifyPairs raises error without train/test split", {
    expect_error(classifyPairs(function() { return(NULL) }), "Need data with train/test split!")
})

test_that("classifyPairs respects minimize", {
    fold = data.frame(a=rep.int(0, 10), best=rep.int("bar", 10), foo=rep.int(0, 10), bar=rep.int(1, 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=F)
    res = classifyPairs(classifier=testclassifier, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="bar", score=1) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="bar", score=1) })))
})

test_that("classifyPairs allows combination function", {
    fold = data.frame(foo=c(rep.int(0, 5), rep.int(1, 5)), bar=c(rep.int(1, 5), rep.int(0, 5)), a=rep.int(1, 10), best=rep.int("foo", 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    res = classifyPairs(classifier=testclassifier, d, combinator=othertestclassifier)
    expectedDf = data.frame(algorithm=c("bar", "foo"), score=c(1, 0))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == expectedDf })))
})
