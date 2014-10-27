test_that("classifyPairs classifies", {
    fold = data.frame(a=rep.int(0, 10), foo=c(rep.int(1, 5), rep.int(0, 5)), bar=c(rep.int(0, 5), rep.int(1, 5)), best=c(rep.int("bar", 5), rep.int("foo", 5)))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = classifyPairs(classifier=idtestclassifier, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="foo", score=1) }) })))
})

test_that("classifyPairs returns predictor", {
    fold = data.frame(a=rep.int(0, 10), foo=c(rep.int(1, 5), rep.int(0, 5)), bar=c(rep.int(0, 5), rep.int(1, 5)), best=c(rep.int("bar", 5), rep.int("foo", 5)))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = classifyPairs(classifier=idtestclassifier, d)
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="foo", score=1) })))
})

test_that("classifyPairs raises error without classifier", {
    expect_error(classifyPairs())
})

test_that("classifyPairs raises error without data", {
    expect_error(classifyPairs(testclassifier))
})

test_that("classifyPairs raises error without train/test split", {
    fold = data.frame(a=rep.int(0, 10), b=c(rep.int(1, 5), rep.int(0, 5)), c=c(rep.int(0, 5), rep.int(1, 5)), best=c(rep.int("c", 5), rep.int("b", 5)))
    d = list(data=rbind(fold, fold), features=c("a"), minimize=T, performance=c("b", "c"))
    class(d) = "llama.data"
    expect_error(classifyPairs(testclassifier, d), "Need data with train/test split!")
})

test_that("classifyPairs respects minimize", {
    fold = data.frame(a=rep.int(0, 10), foo=c(rep.int(1, 5), rep.int(0, 5)), bar=c(rep.int(0, 5), rep.int(1, 5)), best=c(rep.int("bar", 5), rep.int("foo", 5)))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=F)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = classifyPairs(classifier=idtestclassifier, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="foo", score=1) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="foo", score=1) })))
})

test_that("classifyPairs allows combination function", {
    fold = data.frame(a=rep.int(0, 10), foo=c(rep.int(1, 5), rep.int(0, 5)), bar=c(rep.int(0, 5), rep.int(1, 5)), best=c(rep.int("bar", 5), rep.int("foo", 5)))
    d = list(data=rbind(fold, fold), train=list(fold), test=list(fold), features=c("a"), performance=c("foo", "bar"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = classifyPairs(classifier=idtestclassifier, d, combinator=bartestclassifier)
    expectedDf = data.frame(algorithm=c("bar", "foo"), score=c(1, 0))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == expectedDf })))
})
