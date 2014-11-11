limit = as.numeric(Sys.getenv("_R_CHECK_TIMINGS_"))
if(is.na(limit) || limit == 0) {

data(satsolvers)

vbsp = sum(parscores(satsolvers, vbs))
vbsm = sum(misclassificationPenalties(satsolvers, vbs))
vbss = sum(successes(satsolvers, vbs))

test_that("singleBest and vbs", {
    vbsse = sum(apply(satsolvers$data[satsolvers$success], 1, max))
    expect_equal(vbsse, 2125)
    expect_equal(vbss, 2125)

    vbsp1 = sum(parscores(satsolvers, vbs, 1))
    vbsp1e = sum(apply(satsolvers$data[satsolvers$performance], 1, min))
    expect_equal(vbsp1e, 1288664.971)
    expect_equal(vbsp1, 1288664.971)

    expect_equal(vbsp, 11267864.97)
    expect_equal(vbsm, 0)

    sbp = sum(parscores(satsolvers, singleBest))
    sbm = sum(misclassificationPenalties(satsolvers, singleBest))
    sbs = sum(successes(satsolvers, singleBest))

    sbse = sum(satsolvers$data[,"clasp_success"])
    expect_equal(sbse, 2048)
    expect_equal(sbs, 2048)

    sbp1 = sum(parscores(satsolvers, singleBest, 1))
    sbp1e = sum(satsolvers$data["clasp"])
    expect_equal(sbp1e, 1586266.044)
    expect_equal(sbp1, 1586266.044)

    sbme = sum(apply(satsolvers$data[satsolvers$performance], 1, function(x) { abs(x["clasp"] - min(x)) }))
    expect_equal(sbme, 297601.073)
    expect_equal(sbm, 297601.073)

    expect_equal(sbp, 14060266.04)
})

folds = cvFolds(satsolvers)

test_that("classify", {
    res = classify(classifier=makeLearner("classif.OneR"), data=folds)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = classify(classifier=list(makeLearner("classif.OneR"),
                                   makeLearner("classif.OneR"),
                                   makeLearner("classif.OneR")),
                    data=folds)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = classify(classifier=list(makeLearner("classif.OneR"),
                                   makeLearner("classif.OneR"),
                                   makeLearner("classif.OneR"),
                                    .combine=makeLearner("classif.OneR")),
                    data=folds)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))
})

test_that("classifyPairs", {
    res = classifyPairs(classifier=makeLearner("classif.OneR"), data=folds)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = classifyPairs(classifier=makeLearner("classif.OneR"),
                        data=folds,
                        combinator=makeLearner("classif.OneR"))
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))
})

test_that("cluster", {
    res = cluster(clusterer=makeLearner("cluster.SimpleKMeans"), data=folds, pre=normalize)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = cluster(clusterer=makeLearner("cluster.SimpleKMeans"), data=folds,
        bestBy="successes", pre=normalize)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = cluster(clusterer=list(makeLearner("cluster.SimpleKMeans"),
                                 makeLearner("cluster.SimpleKMeans"),
                                 makeLearner("cluster.SimpleKMeans")),
        data=folds, pre=normalize)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = cluster(clusterer=list(makeLearner("cluster.SimpleKMeans"),
                                 makeLearner("cluster.SimpleKMeans"),
                                 makeLearner("cluster.SimpleKMeans"),
        .combine=makeLearner("classif.OneR")), data=folds, pre=normalize)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))
})

test_that("regression", {
    res = regression(regressor=makeLearner("regr.lm"), data=folds)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = regression(regressor=makeLearner("regr.lm"),
                     data=folds,
                     combine=makeLearner("classif.OneR"))
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = regression(regressor=makeLearner("regr.lm"),
                     data=folds,
                     combine=makeLearner("classif.OneR"),
                     expand=function(x) { cbind(x, combn(c(1:ncol(x)), 2,
                            function(y) { abs(x[,y[1]] - x[,y[2]]) })) })
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))
})

test_that("regressionPairs", {
    res = regressionPairs(regressor=makeLearner("regr.lm"), data=folds)
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))

    res = regressionPairs(regressor=makeLearner("regr.lm"), data=folds,
        combine=makeLearner("classif.OneR"))
    expect_true(sum(parscores(folds, res)) > vbsp)
    expect_true(sum(misclassificationPenalties(folds, res)) > vbsm)
    expect_true(sum(successes(folds, res)) < vbss)
    expect_true(is.data.frame(res$predictor(satsolvers$data[satsolvers$features])))
})

}
