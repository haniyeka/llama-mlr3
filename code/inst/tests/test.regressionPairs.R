testregressor = setClass("testregressor", representation(data="list"), contains="formula", where=globalenv())
predict.testregressor = function(object, y) {
    suppressWarnings({
    terms = terms(as.formula(object), data=get("data", envir=environment(object)))
    predictions = eval(attr(terms, "variables")[[2]], envir=environment(object))
    })
    return(rep.int(predictions[1], nrow(y)))
    } 
setMethod("predict", signature(object="testregressor"), predict.testregressor, where=globalenv())

testclassifier = setClass("testclassifier", representation(data="list"), contains="formula", where=globalenv())
predict.testclassifier = function(object, y) { return(rep.int("a", nrow(y))) } 
setMethod("predict", signature(object="testclassifier"), predict.testclassifier, where=globalenv())

test_that("regressionPairs predicts", {
    fold = data.frame(a=rep.int(0, 10), best=rep.int("foo", 10), foo=rep.int(0, 10), bar=rep.int(1, 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    res = regressionPairs(regressor=testregressor, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm=c("foo", "bar"), score=c(1, -1)) }) })))
})

test_that("regressionPairs returns predictor", {
   fold = data.frame(a=rep.int(0, 10), best=rep.int("foo", 10), foo=rep.int(0, 10), bar=rep.int(1, 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    res = regressionPairs(regressor=testregressor, d)
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm=c("foo", "bar"), score=c(1, -1)) })))
})

test_that("regressionPairs raises error without regressor", {
    expect_error(regressionPairs(), "No regressor given!")
})

test_that("regressionPairs raises error without train/test split", {
    expect_error(regressionPairs(function() { return(NULL) }), "Need data with train/test split!")
})

test_that("regressionPairs respects minimize", {
    fold = data.frame(a=rep.int(0, 10), best=rep.int("bar", 10), foo=rep.int(0, 10), bar=rep.int(1, 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=F)
    res = regressionPairs(regressor=testregressor, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm=c("bar", "foo"), score=c(1, -1)) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm=c("bar", "foo"), score=c(1, -1)) })))
})

test_that("regressionPairs allows combine classifier", {
    fold = data.frame(a=rep.int(0, 10), best=rep.int("foo", 10), foo=rep.int(0, 10), bar=rep.int(1, 10))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    res = regressionPairs(regressor=testregressor, d, combine=testclassifier)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="a", score=1) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="a", score=1) })))
})
