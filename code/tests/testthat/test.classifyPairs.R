test_that("classifyPairs classifies", {
    res = classifyPairs(classifier=idtestclassifier, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=1) }) })))
})

test_that("classifyPairs returns predictor", {
    res = classifyPairs(classifier=idtestclassifier, d)
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=1) })))
})

test_that("classifyPairs raises error without classifier", {
    expect_error(classifyPairs())
})

test_that("classifyPairs raises error without data", {
    expect_error(classifyPairs(testclassifier))
})

test_that("classifyPairs raises error without train/test split", {
    expect_error(classifyPairs(testclassifier, dnosplit), "Need data with train/test split!")
})

test_that("classifyPairs respects minimize", {
    res = classifyPairs(classifier=idtestclassifier, d)
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == data.frame(algorithm="b", score=1) }) })))
    expect_true(all(sapply(res$predictor(fold), function(x) { x == data.frame(algorithm="b", score=1) })))
})

test_that("classifyPairs allows combination function", {
    res = classifyPairs(classifier=idtestclassifier, e, combinator=othertestclassifier)
    expectedDf = data.frame(algorithm=c("a", "b"), score=c(1, 0))
    expect_true(all(sapply(1:length(res$predictions), function(i) { sapply(res$predictions[[i]], function(x) { x == expectedDf }) })))
    expect_true(all(sapply(res$predictor(folde), function(x) { x == expectedDf })))
})
