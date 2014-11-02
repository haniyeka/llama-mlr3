test_that("regression predicts", {
    res = regression(testregressor, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm=c("c", "b"), score=c(0, 1)) }) })))
})

test_that("regression returns predictor", {
    res = regression(testregressor, d)
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm=c("c", "b"), score=c(0, 1)) })))
})

test_that("regression raises error without regressor", {
    expect_error(regression())
})

test_that("regression raises error without data", {
    expect_error(regression(testregressor))
})

test_that("regression raises error without train/test split", {
    fold = data.frame(a=rep.int(0, 10), b=c(rep.int(1, 5), rep.int(0, 5)), c=c(rep.int(0, 5), rep.int(1, 5)), best=c(rep.int("c", 5), rep.int("b", 5)))
    d = list(data=rbind(fold, fold), features=c("a"), minimize=T, performance=c("b", "c"))
    class(d) = "llama.data"
    expect_error(regression(testregressor, d))
})

test_that("regression allows to combine by max", {
    fold = data.frame(a=rep.int(0, 10), best=rep.int("b", 10), foo=rep.int(2, 10), bar=rep.int(1, 10))
    d = list(data=rbind(fold, fold), train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)), features=c("a"), performance=c("foo", "bar"), minimize=F)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = regression(testregressor, d)
    expectedDf = data.frame(algorithm=c("foo", "bar"), score=c(2, 1))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == expectedDf })))
})

test_that("regression allows stacking", {
    res = regression(testregressor, d, combine=idtestclassifier)
    expectedDf = data.frame(algorithm=c("b", "c"), score=c(1, 0))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == expectedDf })))
})
