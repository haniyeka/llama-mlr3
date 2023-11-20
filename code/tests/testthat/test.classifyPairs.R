test_that("classifyPairs classifies", {
  set.seed(123)
  res = classifyPairs(classifier=idtestclassifier, data = d)
  expect_equal(unique(res$predictions$id), 11:20)
  expect_equal(res$predictions$algorithm, factor(c("b", "c", "b", "c", "b", "c", "b", "c", "c", "b", "b", "c", "c", "b", "c", "b", "c", "b", "b", "c")))
  expect_equal(res$predictions$score, rep(c(1, 0),10))
})

test_that("classifyPairs returns predictor", {
  set.seed(123)  
  res = classifyPairs(classifier=idtestclassifier, d)
  fold$id = 1:10
  set.seed(123)
  preds = res$predictor(fold)
  expect_equal(unique(preds$id), 1:10)
  expect_equal(preds$algorithm, factor(c("b", "c", "b", "c", "b", "c", "b", "c", "c", "b", "b", "c", "c", "b", "c", "b", "c", "b", "b", "c")))
  expect_equal(preds$score, rep(c(1, 0),10))
})

test_that("classifyPairs returns predictor that works without IDs", {
  set.seed(123)  
  res = classifyPairs(classifier=idtestclassifier, d)
  fold$id = 1:10
  set.seed(123)
  preds = res$predictor(fold[d$features])
  expect_equal(unique(preds$id), 1:10)
  expect_equal(preds$algorithm, factor(c("b", "c", "b", "c", "b", "c", "b", "c", "c", "b", "b", "c", "c", "b", "c", "b", "c", "b", "b", "c")))
  expect_equal(preds$score, rep(c(1, 0),10))
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

test_that("classifyPairs works with three algorithms", {
  set.seed(123)  
  res = classifyPairs(classifier=idtestclassifier, d.three)
  expect_equal(unique(res$predictions$id), 11:20)
  expect_equal(res$predictions$algorithm, factor(c("b", "c", "d", "b", "d", "c", "d", "b", "c", "b", "c", "d", "c", "b", "d", "b", "c", "d", "c", "b", "d", "d", "c", "b", "b", "c", "d", "b", "c", "d")))
  expect_equal(res$predictions$score, c(1, 1, 1, 2, 1, 0, 2, 1, 0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2, 1, 0, 2, 1, 0, 1, 1, 1, 2, 1, 0))
  
  fold$id = 1:10
  set.seed(123)
  preds = res$predictor(fold)
  expect_equal(unique(preds$id), 1:10)
  expect_equal(res$predictions$algorithm, factor(c("b", "c", "d", "b", "d", "c", "d", "b", "c", "b", "c", "d", "c", "b", "d", "b", "c", "d", "c", "b", "d", "d", "c", "b", "b", "c", "d", "b", "c", "d")))
  expect_equal(res$predictions$score, c(1, 1, 1, 2, 1, 0, 2, 1, 0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2, 1, 0, 2, 1, 0, 1, 1, 1, 2, 1, 0))
})

test_that("classifyPairs respects minimize", {
  set.seed(123)
  res = classifyPairs(classifier=idtestclassifier, d)
  expect_equal(unique(res$predictions$id), 11:20)
  expect_equal(res$predictions$algorithm, factor(c("b", "c", "b", "c", "b", "c", "b", "c", "c", "b", "b", "c", "c", "b", "c", "b", "c", "b", "b", "c")))
  expect_equal(res$predictions$score, rep(c(1, 0),10))
  
  fold$id = 1:10
  set.seed(123)
  preds = res$predictor(fold)
  expect_equal(unique(preds$id), 1:10)
  expect_equal(preds$algorithm, factor(c("b", "c", "b", "c", "b", "c", "b", "c", "c", "b", "b", "c", "c", "b", "c", "b", "c", "b", "b", "c")))
  expect_equal(preds$score, rep(c(1, 0),10))
})

test_that("classifyPairs allows combination function", {
  set.seed(123)
  res = classifyPairs(classifier=idtestclassifier, e, combine=othertestclassifier)
  expect_equal(unique(res$predictions$id), 11:20)
  expect_equal(res$predictions$algorithm, factor(c("a", "b", "b", "a", "a", "b", "a", "b", "a", "b", "a", "b", "b", "a", "b", "a", "a", "b", "b", "a")))
  expect_equal(res$predictions$score, rep(c(1, 0),10))
  
  fold$id = 1:10
  set.seed(123)
  preds = res$predictor(fold)
  expect_equal(unique(preds$id), 1:10)
  expect_equal(preds$algorithm, factor(c("b", "a", "b", "a", "b", "a", "a", "b", "b", "a", "a", "b", "b", "a", "a", "b", "a", "b", "a", "b")))
  expect_equal(preds$score, rep(c(1, 0),10))
})

# test_that("classifyPairs works with probabilities", {
#   res = classifyPairs(classifier=probtestclassifier , d)
#   expect_equal(unique(res$predictions$id), 11:20)
#   by(res$predictions, res$predictions$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("c", "b")))
#     expect_equal(ss$score, c(.2, .1))
#   })
#   fold$id = 1:10
#   preds = res$predictor(fold)
#   expect_equal(unique(preds$id), 1:10)
#   by(preds, preds$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("c", "b")))
#     expect_equal(ss$score, c(.2, .1))
#   })
#   res = classifyPairs(classifier=probtestclassifier, combine=probtestclassifier, d)
#   expect_equal(unique(res$predictions$id), 11:20)
#   by(res$predictions, res$predictions$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("c", "b")))
#     expect_equal(ss$score, c(.2, .1))
#   })
#   fold$id = 1:10
#   preds = res$predictor(fold)
#   expect_equal(unique(preds$id), 1:10)
#   by(preds, preds$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("c", "b")))
#     expect_equal(ss$score, c(.2, .1))
#   })
# })
