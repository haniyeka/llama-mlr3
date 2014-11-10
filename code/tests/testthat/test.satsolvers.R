data(satsolvers)
folds = cvFolds(satsolvers)

test_that("classify", {
    res = classify(classifier=makeLearner("classif.J48"), data=folds)
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = classify(classifier=list(makeLearner("classif.J48"),
                                   makeLearner("classif.IBk"),
                                   makeLearner("classif.svm")),
                    data=folds)
    expect_true(is.numeric(sum(parscores(folds, res))))
    res = classify(classifier=list(makeLearner("classif.J48"),
                                   makeLearner("classif.IBk"),
                                   makeLearner("classif.svm"),
                                    .combine=makeLearner("classif.J48")),
                    data=folds)
    expect_true(is.numeric(sum(parscores(folds, res))))
})

test_that("classifyPairs", {
    res = classifyPairs(classifier=makeLearner("classif.J48"), data=folds)
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = classifyPairs(classifier=makeLearner("classif.J48"),
                        data=folds,
                        combinator=makeLearner("classif.J48"))
    expect_true(is.numeric(sum(parscores(folds, res))))
})

test_that("cluster", {
    res = cluster(clusterer=makeLearner("cluster.XMeans"), data=folds, pre=normalize)
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = cluster(clusterer=makeLearner("cluster.XMeans"), data=folds,
        bestBy="successes", pre=normalize)
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = cluster(clusterer=list(makeLearner("cluster.XMeans"),
        makeLearner("cluster.SimpleKMeans"), makeLearner("cluster.EM")),
        data=folds, pre=normalize)
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = cluster(clusterer=list(makeLearner("cluster.XMeans"),
        makeLearner("cluster.SimpleKMeans"), makeLearner("cluster.EM"),
        .combine=makeLearner("classif.J48")), data=folds, pre=normalize)
    expect_true(is.numeric(sum(parscores(folds, res))))
})

test_that("regression", {
    res = regression(regressor=makeLearner("regr.lm"), data=folds)
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = regression(regressor=makeLearner("regr.ksvm"), data=folds)
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = regression(regressor=makeLearner("regr.ksvm"),
                     data=folds,
                     combine=makeLearner("classif.J48"))
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = regression(regressor=makeLearner("regr.ksvm"),
                     data=folds,
                     combine=makeLearner("classif.J48"),
                     expand=function(x) { cbind(x, combn(c(1:ncol(x)), 2,
                            function(y) { abs(x[,y[1]] - x[,y[2]]) })) })
    expect_true(is.numeric(sum(parscores(folds, res))))
})

test_that("regressionPairs", {
    res = regressionPairs(regressor=makeLearner("regr.lm"), data=folds)
    expect_true(is.numeric(sum(parscores(folds, res))))

    res = regressionPairs(regressor=makeLearner("regr.lm"), data=folds,
        combine=makeLearner("classif.J48"))
    expect_true(is.numeric(sum(parscores(folds, res))))
})
