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
    expect_error(cluster(clusterer = testclusterer, data = d, bestBy="foo"), "Unknown bestBy: foo")
})

test_that("cluster finds best for cluster", {
  res = cluster(testclusterer, g)
  algs = c("c", "b")
  expect_equal(unique(res$predictions$id), 11:20)
  by(res$predictions, res$predictions$id, function(ss) {
    expect_equal(ss$algorithm, factor(algs, levels=algs))
    expect_equal(ss$score, c(0, 1))
  })
})

test_that("cluster finds best by count for cluster", {
    res = cluster(testclusterer, g, bestBy="count")
    expect_equal(unique(res$predictions$id), 11:20)
    by(res$predictions, res$predictions$id, function(ss) {
        expect_equal(ss$algorithm, factor("b"))
        expect_equal(ss$score, 10)
    })
})

test_that("cluster finds best by successes for cluster", {
    res = cluster(testclusterer, g, bestBy="successes")
    expect_equal(unique(res$predictions$id), 11:20)
    by(res$predictions, res$predictions$id, function(ss) {
        expect_equal(ss$algorithm, factor(c("b", "c")))
        expect_equal(ss$score, c(1, 0))
    })
})

test_that("cluster returns predictor", {
    res = cluster(testclusterer, g)
    algs = c("c", "b")
    foldg$id = 1:10
    preds = res$predictor(foldg)
    expect_equal(unique(preds$id), 1:10)
    by(preds, preds$id, function(ss) {
        expect_equal(ss$algorithm, factor(algs, levels=algs))
        expect_equal(ss$score, c(0, 1))
    })
})

test_that("cluster returns predictor that works without IDs", {
    res = cluster(testclusterer, g)
    algs = c("c", "b")
    foldg$id = 1:10
    preds = res$predictor(foldg[g$features])
    expect_equal(unique(preds$id), 1:10)
    by(preds, preds$id, function(ss) {
        expect_equal(ss$algorithm, factor(algs, levels=algs))
        expect_equal(ss$score, c(0, 1))
    })
})

test_that("cluster takes list of clusterers", {
    res = cluster(list(testclusterer, testclusterer, testclusterer), g)
    algs = c("c", "b")
    expect_equal(unique(res$predictions$id), 11:20)
    by(res$predictions, res$predictions$id, function(ss) {
        expect_equal(ss$algorithm, factor(algs, levels=algs))
        expect_equal(ss$score, c(0, 3))
    })

    foldg$id = 1:10
    preds = res$predictor(foldg)
    expect_equal(unique(preds$id), 1:10)
    by(preds, preds$id, function(ss) {
        expect_equal(ss$algorithm, factor(algs, levels=algs))
        expect_equal(ss$score, c(0, 3))
    })
})

test_that("cluster takes list of clusterers and combinator", {
    res = cluster(clusterer = list(testclusterer, testclusterer, testclusterer, .combine=idtestclassifier), g)
    expect_equal(unique(res$predictions$id), 11:20)
    algs = c("b", "c")
    by(res$predictions, res$predictions$id, function(ss) {
        expect_equal(ss$algorithm, factor(algs, levels=algs))
        expect_equal(ss$score, c(0, 1))
    })

    foldg$id = 1:10
    preds = res$predictor(foldg)
    expect_equal(unique(preds$id), 1:10)
    by(preds, preds$id, function(ss) {
      expect_equal(ss$algorithm, factor(algs, levels=algs))
      expect_equal(ss$score, c(0, 1))
    })
})
