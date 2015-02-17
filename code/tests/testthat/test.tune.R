test_that("tuneModel tunes", {
    ps = makeParamSet(makeIntegerParam("test", lower = 1, upper = 100))
    design = generateRandomDesign(10, ps)
    res = tuneModel(d, classify, testclassifier, design, misclassificationPenalties, nfolds = 2L, quiet = TRUE)

    expect_equal(length(res), 1)
    expect_true(is.data.frame(res[[1]]$parameters))
    expect_equal(res[[1]]$score, 0.5)
    expect_equal(class(res[[1]]$model), "llama.model")
})
