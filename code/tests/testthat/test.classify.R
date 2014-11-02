test_that("classify classifies", {
    res = classify(classifier=testclassifier, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=1) }) })))
})

test_that("classify returns predictor", {
    res = classify(classifier=testclassifier, d)
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=1) })))
})

test_that("classify raises error without classifier", {
    expect_error(classify())
})

test_that("classify raises error without data", {
    expect_error(classify(testclassifier))
})

test_that("classify raises error without train/test split", {
    expect_error(classify(testclassifier, dnosplit), "Need data with train/test split!")
})

test_that("classify takes list of classifiers", {
    res = classify(classifier=list(testclassifier, testclassifier, testclassifier), d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=3) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=3) })))
})

test_that("classify takes list of classifiers and combination function", {
    res = classify(classifier=list(foo, foo, foo, .combine=othertestclassifier), e)
    expectedDf = data.frame(algorithm=c("a", "b"), score=c(1, 0))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(folde), function(x) { x == expectedDf })))
})

test_that("classify ensemble does majority voting by default", {
    res = classify(classifier=list(testclassifier, othertestclassifier, othertestclassifier), e)
    expectedDf = data.frame(algorithm=c("a", "b"), score=c(2, 1))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(folde), function(x) { x == expectedDf })))
})
