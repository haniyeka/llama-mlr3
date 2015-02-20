test_that("tuneModel tunes", {
    ps = makeParamSet(makeIntegerParam("test", lower = 1, upper = 100))
    design = generateRandomDesign(10, ps)
    res = tuneModel(d, classify, testclassifier, design, misclassificationPenalties, nfolds = 2L, quiet = TRUE)

    expect_true(is.data.frame(res))
    expect_equal(dim(res), c(10, 4))
    #expect_true(all(res$inner.score == 1.25))
    expect_true(all(res$outer.score == 1))
})
