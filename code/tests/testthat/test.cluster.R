test_that("cluster raises error without clusterer", {
    expect_error(cluster())
})

test_that("cluster raises error without data", {
    expect_error(cluster(testclusterer))
})

test_that("cluster raises error without train/test split", {
    fold = data.frame(a=rep.int(0, 10), b=c(rep.int(1, 5), rep.int(0, 5)), c=c(rep.int(0, 5), rep.int(1, 5)), best=c(rep.int("c", 5), rep.int("b", 5)))
    d = list(data=rbind(fold, fold), features=c("a"), minimize=T, performance=c("b", "c"))
    class(d) = "llama.data"
    expect_error(cluster(testclusterer, d))
})

test_that("cluster raises error with unknown bestBy", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)), features=c("a"), performance=c("b"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    expect_error(cluster(testclusterer, d, bestBy="foo"), "Unknown bestBy: foo")
})

test_that("cluster finds best for cluster", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), c=rep.int(0, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)), features=c("a"), performance=c("b", "c"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = cluster(testclusterer, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm=c("c", "b"), score=c(0, 10)) }) })))
})

test_that("cluster finds best by count for cluster", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), c=rep.int(0, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)), features=c("a"), performance=c("b", "c"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = cluster(testclusterer, d, bestBy="count")
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=10) }) })))
})

test_that("cluster finds best by successes for cluster", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), c=rep.int(0, 10), best=rep.int("b", 10), d=rep.int(T, 10), e=rep.int(F, 10))
    d = list(data=rbind(fold, fold), train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)), features=c("a"), performance=c("b", "c"), success=c("d", "e"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = cluster(testclusterer, d, bestBy="successes")
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm=c("b", "c"), score=c(10, 0)) }) })))
})

test_that("cluster returns predictor", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)), features=c("a"), performance=c("b"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = cluster(testclusterer, d)
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=20) })))
})

test_that("cluster takes list of clusterers", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)), features=c("a"), performance=c("b"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = cluster(list(testclusterer, testclusterer, testclusterer), d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=30) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=60) })))
})

test_that("cluster takes list of clusterers and combinator", {
    fold = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), best=rep.int("b", 10))
    d = list(data=rbind(fold, fold), train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)), features=c("a"), performance=c("b"), minimize=T)
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = cluster(list(testclusterer, testclusterer, testclusterer, .combine=testclassifier), d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=1) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=1) })))
})
