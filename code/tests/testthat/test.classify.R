test_that("Input validation for classifier and data parameters in classify function", {
  # NULL classifier
  expect_error(classify(classifier = NULL, data = d), 
               "Need classifier or list of classifiers!")
  # NULL data
  expect_error(classify(classifier = testclassifier, data = NULL), 
               "Must inherit from class 'llama.data', but has class 'NULL'")
  
  # correct data and classifier
  expect_silent(classify(classifier = testclassifier, data = d))
  # list of classifiers 
  expect_silent(classify(classifier = list(testclassifier, testclassifier), data = d))
  
  # NA save.models
  expect_silent(classify(classifier = testclassifier, data = d, save.models = NA))
  
  # use weights
  expect_silent(classify(classifier = testclassifier, data = d, use.weights = TRUE))
  # don't use weights
  expect_silent(classify(classifier = testclassifier, data = d, use.weights = FALSE))
  # non boolean weights
  expect_error(classify(classifier = testclassifier, data = d, use.weights = "not a boolean"))
})

test_that("Functionality test",{
  # correct data and classifier
  expect_silent(classify(classifier = testclassifier, data = d))
  # list of classifiers 
  expect_silent(classify(classifier = list(testclassifier, testclassifier), data = d))
  
})

test_that("classify classifies",{
  set.seed(123)
  res = classify(classifier = testclassifier, data = d)
  algs = c("b", "c", "b", "c", "b", "c", "c", "b", "b", "c", "c", "b", "c", "b", "c", "b", "b", "c", "b", "c")
  scores = rep(c(1, 0),10)
  expect_equal(res$predictions$algorithm, factor(algs))
  expect_equal(res$predictions$score, scores)
})

test_that("classify returns predictor",{
  set.seed(123)
  res = classify(classifier = testclassifier, data = d)
  fold$id = 1:10
  
  set.seed(123)
  preds = res$predictor(fold)
  expect_equal(unique(preds$id), 1:10)
  
  algs = c("b", "c", "b", "c", "b", "c", "c", "b", "b", "c", "c", "b", "c", "b", "c", "b", "b", "c", "b", "c")
  scores = rep(c(1, 0),10)
  expect_equal(preds$algorithm, factor(algs))
  expect_equal(preds$score, scores)
})

test_that("classify returns predictor that works without IDs",{
  set.seed(123)
  res = classify(classifier = testclassifier, data = d)
  fold$id = 1:10
  
  set.seed(123)
  preds = res$predictor(fold[d$features])
  expect_equal(unique(preds$id), 1:10)
  
  algs = c("b", "c", "b", "c", "b", "c", "c", "b", "b", "c", "c", "b", "c", "b", "c", "b", "b", "c", "b", "c")
  scores = rep(c(1, 0),10)
  expect_equal(preds$algorithm, factor(algs))
  expect_equal(preds$score, scores)
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
  set.seed(123)
  res = classify(classifier=list(testclassifier, testclassifier, testclassifier), d)
  expect_equal(unique(res$predictions$id), 11:20)
  
  
  algs = c("b", "c", "c", "b", "b", "c", "b", "c", "b", "c", "b", "c", "c", "b", "c", "b", "b", "c", "b", "c")
  scores = c(2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1)
  expect_equal(res$predictions$algorithm, factor(algs))
  expect_equal(res$predictions$score, scores)
  
  fold$id = 1:10
  set.seed(123)
  preds = res$predictor(fold)
  
  expect_equal(unique(preds$id), 1:10)
  expect_equal(preds$algorithm, factor(algs))
  expect_equal(preds$score, scores)
})

test_that("classify takes list of classifiers and combination function", {
  set.seed(123)
  res = classify(classifier=list(foo, foo, foo, .combine=othertestclassifier), e)
  expect_equal(unique(res$predictions$id), 11:20)
  
  algs = c("a", "b", "b", "a", "b", "a", "a", "b", "a", "b", "b", "a", "a", "b", "a", "b", "a", "b", "a", "b")
  scores = rep(c(1,0),10)
  expect_equal(res$predictions$algorithm, factor(algs))
  expect_equal(res$predictions$score, scores)
  
  folde$id = 1:10
  set.seed(123)
  preds = res$predictor(folde)
  algs = c("a", "b", "b", "a", "a", "b", "b", "a", "b", "a", "a", "b", "a", "b", "a", "b", "a", "b", "b", "a")

  expect_equal(unique(preds$id), 1:10)
  expect_equal(preds$algorithm, factor(algs))
  expect_equal(preds$score, scores)
})

test_that("classify ensemble does majority voting by default", {
  set.seed(123)
  res = classify(classifier=list(testclassifier, othertestclassifier, othertestclassifier), e)
  expect_equal(unique(res$predictions$id), 11:20)
  
  algs = c("a", "b", "b", "a", "a", "b", "a", "b", "a", "b", "a", "b", "b", "a", "b", "a", "a", "b", "a", "b")
  scores = c(2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1)
  expect_equal(res$predictions$algorithm, factor(algs))
  expect_equal(res$predictions$score, scores)
  
  folde$id = 1:10
  set.seed(123)
  preds = res$predictor(folde)

  expect_equal(unique(preds$id), 1:10)
  expect_equal(preds$algorithm, factor(algs))
  expect_equal(preds$score, scores)
  
})

# test_that("classify works with probabilities", {
#   set.seed(123)
#   res = classify(classifier=probtestclassifier, data = d)
#   expect_equal(unique(res$predictions$id), 11:20)
#     by(res$predictions, res$predictions$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("b", "c")))
#     expect_equal(ss$score, c(.5, .5))
#   })
#   
#   fold$id = 1:10
#   set.seed(123)
#   preds = res$predictor(fold)
#   expect_equal(unique(preds$id), 1:10)
#   by(preds, preds$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("b", "c")))
#     expect_equal(ss$score, c(.5, .5))
#   })
#   
#   set.seed(123)
#   res = classify(classifier=list(probtestclassifier, probtestclassifier, probtestclassifier), d)
#   expect_equal(unique(res$predictions$id), 11:20)
#   by(res$predictions, res$predictions$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("b", "c")))
#     expect_equal(ss$score, c(1.5, 1.5))
#   })
#   
#   fold$id = 1:10
#   set.seed(123)
#   preds = res$predictor(fold)
#   expect_equal(unique(preds$id), 1:10)
#   by(preds, preds$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("b", "c")))
#     expect_equal(ss$score, c(1.5, 1.5))
#   })
#   
#   set.seed(123)
#   res = classify(classifier=list(probtestclassifier, probtestclassifier, probtestclassifier, .combine = probtestclassifier), d)
#   expect_equal(unique(res$predictions$id), 11:20)
#   by(res$predictions, res$predictions$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("b", "c")))
#     expect_equal(ss$score, c(.5, .5))
#   })
#   
#   fold$id = 1:10
#   set.seed(123)
#   preds = res$predictor(fold)
#   expect_equal(unique(preds$id), 1:10)
#   by(preds, preds$id, function(ss) {
#     expect_equal(ss$algorithm, factor(c("c", "b")))
#     expect_equal(ss$score, c(.2, .1))
#   })
# })

