test_that("cluster raises error without clusterer", {
    expect_error(cluster())
})

test_that("cluster raises error without data", {
    expect_error(cluster(testclusterer))
})

test_that("cluster raises error without train/test split", {
    expect_error(cluster(testclusterer, dnosplit))
})

test_that("cluster raises error with unknown bestBy", {
    expect_error(cluster(testclusterer, d, bestBy="foo"), "Unknown bestBy: foo")
})

test_that("cluster finds best for cluster", {
    res = cluster(testclusterer, g)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm=c("c", "b"), score=c(0, 10)) }) })))
})

test_that("cluster finds best by count for cluster", {
    res = cluster(testclusterer, g, bestBy="count")
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=10) }) })))
})

test_that("cluster finds best by successes for cluster", {
    res = cluster(testclusterer, g, bestBy="successes")
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm=c("b", "c"), score=c(10, 0)) }) })))
})

test_that("cluster returns predictor", {
    res = cluster(testclusterer, g)
    expect_true(all(sapply(res$predictor(foldg), function(x) { x == data.frame(algorithm=c("c", "b"), score=c(0, 20)) })))
})

test_that("cluster takes list of clusterers", {
    res = cluster(list(testclusterer, testclusterer, testclusterer), g)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm=c("c", "b"), score=c(0, 30)) }) })))
    expect_true(all(sapply(res$predictor(foldg), function(x) { x == data.frame(algorithm=c("c", "b"), score=c(0, 60)) })))
})

test_that("cluster takes list of clusterers and combinator", {
    res = cluster(list(testclusterer, testclusterer, testclusterer, .combine=idtestclassifier), g)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="c", score=1) }) })))
    expect_true(all(sapply(res$predictor(foldg), function(x) { x == data.frame(algorithm="c", score=1) })))
})
