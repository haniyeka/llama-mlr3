test_that("Input validation for regressor and data parameters in regressionPairs function", {
  # NULL regressor
  expect_error(regressionPairs(regressor = NULL, data = d), 
               "Need regressor!")
  # NULL data
  expect_error(regressionPairs(regressor = testregressor, data = NULL), 
               "Must inherit from class 'llama.data', but has class 'NULL'")
  
  # correct data and regressor
  expect_silent(regressionPairs(regressor = testregressor, data = d))
  
  # NA save.models
  expect_silent(regressionPairs(regressor = testregressor, data = d, save.models = NA))
  
  # use weights
  expect_silent(regressionPairs(regressor = testregressor, data = d, use.weights = TRUE))
  # don't use weights
  expect_silent(regressionPairs(regressor = testregressor, data = d, use.weights = FALSE))
})

test_that("regressionPairs predicts", {
  set.seed(123)
  res = regressionPairs(regressor=testregressor, d)
  expect_equal(unique(res$predictions$id), 11:20)
  algs = rep(c("b", "c"),10)
  scores = rep(c(0, 0),10)
  expect_equal(res$predictions$algorithm, factor(algs))
  expect_equal(res$predictions$score, scores)
  
  # same test with algorithm features
  res.algo = regressionPairs(regressor=algotestregressor, d.algo)
  expect_equal(unique(res.algo$predictions$id), 11:20)
  expect_equal(res$predictions$algorithm, factor(algs))
  expect_equal(res$predictions$score, scores)
})

test_that("regressionPairs returns predictor", {
  set.seed(123)
  res = regressionPairs(regressor=testregressor, d)
  fold$id = 1:10
  preds = res$predictor(fold)
  expect_equal(unique(preds$id), 1:10)
  by(preds, preds$id, function(ss) {
      expect_equal(ss$algorithm, factor(c("b", "c")))
      expect_equal(ss$score, c(0, 0))
  })
    
  # same test with algorithm features
  res.algo = regressionPairs(regressor=algotestregressor, d.algo)
  fold.algo$id = rep.int(1:10, rep.int(2, 10))
  preds.algo = res.algo$predictor(fold.algo)
  expect_equal(unique(preds.algo$id), 1:10)
  by(preds.algo, preds.algo$id, function(ss) {
      stopifnot(levels(ss$algorithm) %in% c("c", "b"))
      expect_equal(ss$score, c(0, 0))
  })
})

test_that("regressionPairs raises error without regressor", {
    expect_error(regressionPairs())
})

test_that("regressionPairs raises error without data", {
    expect_error(regressionPairs(testregressor))
})

test_that("regressionPairs raises error without train/test split", {
    fold = data.frame(a=rep.int(0, 10), b=c(rep.int(1, 5), rep.int(0, 5)), c=c(rep.int(0, 5), rep.int(1, 5)), best=c(rep.int("c", 5), rep.int("b", 5)))
    d = list(data=rbind(fold, fold), features=c("a"), minimize=T, performance=c("b", "c"))
    class(d) = "llama.data"
    expect_error(regressionPairs(testregressor, d))
    
    # same test with algorithm features
    fold.algo = data.frame(a=rep.int(0, 20), p=c(rep.int(c(1, 0), 5), rep.int(c(0, 1), 5)), best=c(rep.int("c", 5), rep.int("b", 5)), 
                           algo=rep.int(c("b", "c"), 10), s=rep.int(c(2, 3), 10))
    d.algo = list(data=rbind(fold.algo, fold.algo), features=c("a"), minimize=T, performance=c("p"),
             algos=c("a"), algorithmFeatures=c("s"))
    class(d.algo) = "llama.data"
    expect_error(regressionPairs(algotestregressor, d.algo))
})

test_that("regressionPairs works with three algorithms", {
  set.seed(123)
  res = regressionPairs(regressor=testregressor, d.three)
  expect_equal(unique(res$predictions$id), 11:20)
  by(res$predictions, res$predictions$id, function(ss) {
    expect_equal(ss$algorithm, factor(c("b", "c", "d")))
    expect_equal(ss$score, c(0.5, 0.5, -1.0))
  })

  fold$id = 1:10
  set.seed(123)
  preds = res$predictor(fold)
  expect_equal(unique(preds$id), 1:10)
  by(preds, preds$id, function(ss) {
    expect_equal(ss$algorithm, factor(c("b", "c", "d")))
    expect_equal(ss$score, c(1, 0, -1.0))
  })
  
  # same test with algorithm features
  res.algo = regressionPairs(regressor=testregressor, d.three.algo)
  expect_equal(unique(res.algo$predictions$id), 11:20)
  by(res.algo$predictions, res.algo$predictions$id, function(ss) {
    stopifnot(levels(ss$algorithm) %in% c("b", "c", "d"))
    expect_equal(ss$score, c(0.6666667, 0, -0.6666667),tolerance = 1e-7)
  })
  
  fold.three.algo$id = rep.int(1:10, rep.int(3, 10))
  preds.algo = res.algo$predictor(fold.three.algo)
  expect_equal(unique(preds.algo$id), 1:10)
  by(preds.algo, preds.algo$id, function(ss) {
    stopifnot(levels(ss$algorithm) %in% c("b", "c", "d"))
    expect_equal(ss$score, c(0.6666667, 0, -0.6666667),tolerance = 1e-7)
  })
})

test_that("regressionPairs respects minimize", {
    fold = data.frame(a=rep.int(0, 10), best=rep.int("b", 10), foo=rep.int(2, 10), bar=rep.int(1, 10))
    d = list(data=rbind(cbind(fold, id=1:10), cbind(fold, id=11:20)),
        train=list(1:nrow(fold)), test=list(1:nrow(fold) + nrow(fold)),
        features=c("a"), performance=c("foo", "bar"), minimize=F, ids=c("id"))
    class(d) = "llama.data"
    attr(d, "hasSplits") = TRUE
    res = regressionPairs(regressor=testregressor, d)
    expect_equal(unique(res$predictions$id), 11:20)
    by(res$predictions, res$predictions$id, function(ss) {
        expect_equal(ss$algorithm, factor(c("foo", "bar")))
        expect_equal(ss$score, c(1, -1))
    })

    fold$id = 1:10
    preds = res$predictor(fold)
    expect_equal(unique(preds$id), 1:10)
    by(preds, preds$id, function(ss) {
        expect_equal(ss$algorithm, factor(c("foo", "bar")))
        expect_equal(ss$score, c(1, -1))
    })
    
    # same test with algorithm features
    fold.algo = data.frame(a=rep.int(0, 20), best=rep.int("b", 10), p=c(rep.int(c(2, 1), 5), rep.int(c(3, 1), 5)), 
                           algo=rep.int(c("foo", "bar"), 10), s=rep.int(c(2, 3), 10))
    d.algo = list(data=rbind(cbind(fold.algo, id=rep.int(1:10, rep.int(2, 10))), cbind(fold.algo, id=rep.int(11:20, rep.int(2, 10)))),
             train=list(1:nrow(fold.algo)), test=list(1:nrow(fold.algo) + nrow(fold.algo)),
             features=c("a"), performance=c("p"), minimize=F, ids=c("id"),
             algorithmFeatures=c("s"), algos=c("algo"), algorithmNames=c("foo", "bar"))
    class(d.algo) = "llama.data"
    attr(d.algo, "hasSplits") = TRUE
    res.algo = regressionPairs(regressor=algotestregressor, d.algo)
    expect_equal(unique(res.algo$predictions$id), 11:20)
    by(res.algo$predictions, res.algo$predictions$id, function(ss) {
        expect_equal(ss$algorithm, factor(c("foo", "bar")))
        expect_true(any(ss$score == c(1.5, -1.5), ss$score == c(2, -2), equals(TRUE)))
    })
    
    fold.algo$id = rep.int(1:10, rep.int(2, 10))
    preds.algo = res.algo$predictor(fold.algo)
    expect_equal(unique(preds.algo$id), 1:10)
    by(preds.algo, preds.algo$id, function(ss) {
        expect_equal(ss$algorithm, factor(c("foo", "bar")))
        expect_true(any(ss$score == c(1.5, -1.5), ss$score == c(2, -2)))
    })
})

test_that("regressionPairs doesn't allow combine classifier with algorithm features", {
    expect_error(regressionPairs(regressor=algotestregressor, d.algo, combine=idtestclassifier))
})

