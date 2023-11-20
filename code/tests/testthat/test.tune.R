test_that("tuneModel tunes", {
    ps = makeParamSet(makeIntegerLearnerParam("num.trees", lower = 1, upper = 20))
    design = generateRandomDesign(10, ps)
    res = tuneModel(ldf = d, llama.fun = classify, learner = tuningtestclassifier, design = design, metric = misclassificationPenalties, nfolds = 2L, quiet = TRUE)

    expect_equal(class(res), "llama.model")
    expect_equal(attr(res, "type"), "classify")

    expect_equal(dim(res$predictions), c(20, 4))
    expect_true(res$parvals$num.trees >= 1 && res$parvals$num.trees <= 20)
    expect_true(all(sapply(res$inner.parvals, function(x) (x$num.trees >= 1 && x$num.trees <= 20))))
})
