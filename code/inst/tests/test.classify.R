makeRLearner.classif.test = function() {
    makeRLearnerClassif(cl = "classif.test", package="llama", par.set=makeParamSet(), numerics=T, factors=T)
}
trainLearner.classif.test = function(.learner, .task, .subset, .weights, ...) { }
predictLearner.classif.test = function(learner, model, newdata) {
    return(factor(rep.int("b", nrow(newdata))))
}
testclassifier = makeLearner("classif.test")

makeRLearner.classif.otest = function() {
    makeRLearnerClassif(cl = "classif.otest", package="llama", par.set=makeParamSet(), numerics=T, factors=T)
}
trainLearner.classif.otest = function(.learner, .task, .subset, .weights, ...) { }
predictLearner.classif.otest = function(learner, model, newdata) {
    return(factor(rep.int("a", nrow(newdata))))
}
othertestclassifier = makeLearner("classif.otest")

makeRLearner.classif.ftest = function() {
    makeRLearnerClassif(cl = "classif.ftest", package="llama", par.set=makeParamSet(), numerics=T, factors=T)
}
trainLearner.classif.ftest = function(.learner, .task, .subset, .weights, ...) { }
predictLearner.classif.ftest = function(learner, model, newdata) {
    return(factor(c(rep.int("a", nrow(newdata)/2), rep.int("b", nrow(newdata)/2))))
}
foo = makeLearner("classif.ftest")

test_that("classify classifies", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), minimize=T, performance=c("b"))
    res = classify(classifier=testclassifier, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=1) }) })))
})

test_that("classify returns predictor", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), minimize=T, performance=c("b"))
    res = classify(classifier=testclassifier, d)
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=1) })))
})

test_that("classify raises error without classifier", {
    expect_error(classify(), "No classifier given!")
})

test_that("classify raises error without train/test split", {
    expect_error(classify(function() { return(NULL) }), "Need data with train/test split!")
})

test_that("classify takes list of classifiers", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), minimize=T, performance=c("b"))
    res = classify(classifier=list(testclassifier, testclassifier, testclassifier), d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=3) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=3) })))
})

test_that("classify takes list of classifiers and combination function", {
    fold = data.frame(a=c(rep.int(0, 5), rep.int(1, 5)), b=c(rep.int(1, 5), rep.int(0, 5)), c=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("c"), minimize=T, performance=c("a", "b"))
    res = classify(classifier=list(foo, foo, foo, .combine=othertestclassifier), d)
    expectedDf = data.frame(algorithm=c("a", "b"), score=c(1, 0))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == expectedDf })))
})

test_that("classify ensemble does majority voting by default", {
    fold = data.frame(a=c(rep.int(0, 5), rep.int(1, 5)), b=c(rep.int(1, 5), rep.int(0, 5)), c=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("c"), minimize=T, performance=c("a", "b"))
    res = classify(classifier=list(testclassifier, othertestclassifier, othertestclassifier), d)
    expectedDf = data.frame(algorithm=c("a", "b"), score=c(2, 1))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == expectedDf })))
})
