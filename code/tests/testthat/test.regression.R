test_that("Input validation for regressor and data parameters in regression function", {
  # NULL regressor
  expect_error(regression(regressor = NULL, data = d), 
               "Need regressor!")
  # NULL data
  expect_error(regression(regressor = testregressor, data = NULL), 
               "Must inherit from class 'llama.data', but has class 'NULL'")
  
  # correct data and regressor
  expect_silent(regression(regressor = testregressor, data = d))
  
  # NA save.models
  expect_silent(regression(regressor = testregressor, data = d, save.models = NA))
  
  # use weights
  expect_silent(regression(regressor = testregressor, data = d, use.weights = TRUE))
  # don't use weights
  expect_silent(regression(regressor = testregressor, data = d, use.weights = FALSE))
})

test_that("Functionality test",{
  # correct data and regressor
  expect_silent(regression(regressor = testregressor, data = d))
})

test_that("regression predicts",{
  set.seed(123)
  res = regression(regressor = testregressor, data = d)
  algs = rep(c("b", "c"),10)
  scores = rep(c(0.5, 0.5),10)
  expect_equal(res$predictions$algorithm, factor(algs))
  expect_equal(res$predictions$score, scores)
})

test_that("regression returns predictor",{
  set.seed(123)
  res = regression(regressor = testregressor, data = d)
  fold$id = 1:10
  
  set.seed(123)
  preds = res$predictor(fold)
  expect_equal(unique(preds$id), 1:10)
  
  algs = rep(c("b", "c"),10)
  scores = rep(c(0.5, 0.5),10)
  expect_equal(preds$algorithm, factor(algs))
  expect_equal(preds$score, scores)
})

test_that("regression returns predictor that works without IDs",{
  set.seed(123)
  res = regression(regressor = testregressor, data = d)
  fold$id = 1:10
  
  set.seed(123)
  preds = res$predictor(fold[d$features])
  expect_equal(unique(preds$id), 1:10)
  
  algs = rep(c("b", "c"),10)
  scores = rep(c(0.5, 0.5),10)
  expect_equal(preds$algorithm, factor(algs))
  expect_equal(preds$score, scores)
})

test_that("regression raises error without regressor", {
  expect_error(regression())
})

test_that("regression raises error without data", {
  expect_error(regression(testregressor))
})

test_that("regression raises error without train/test split", {
  expect_error(regression(testregressor, dnosplit), "Need data with train/test split!")
})

