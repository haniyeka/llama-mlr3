# code adapted from aslib package
tuneModel <-
function(ldf, llama.fun, learner, design, metric = parscores, nfolds = 10L, quiet = FALSE) {
    assertClass(ldf, "llama.data")
    assertClass(llama.fun, "llama.modelFunction")
    assertClass(learner, "Learner")
    assertClass(design, "data.frame")
    assertClass(metric, "llama.metric")
    assertInteger(nfolds)

    if(length(attr(ldf, "hasSplits")) == 0) {
        n.outer.folds = nfolds
        ldf = cvFolds(ldf, nfolds = nfolds)
    } else {
        n.outer.folds = length(ldf$test)
    }
    retval = lapply(1:n.outer.folds, function(i) {
        if(!quiet) message(paste("Fold ", i, "/", n.outer.folds, ":", sep = ""))
        ldf2 = ldf
        ldf2$data = ldf$data[ldf$train[[i]],]
        ldf3 = ldf2
        ldf3 = cvFolds(ldf3, nfolds = nfolds)
        parvals = tuneLlamaModel(ldf3, llama.fun, learner, design, metric, quiet)

        # now fit only on outer trainining set with best params and predict outer test set
        learner2 = setHyperPars(learner, par.vals = as.list(parvals))
        outer.split.ldf = ldf
        outer.split.ldf$train = list(ldf$train[[i]])
        outer.split.ldf$test = list(ldf$test[[i]])
        model = llama.fun(learner2, data = outer.split.ldf)
        score = mean(metric(outer.split.ldf, model))

        return(list(model = model, score = score, parameters = parvals))
    })

    return(retval)
}

tuneLlamaModel <-
function(ldf, llama.fun, learner, design, metric, quiet) {
    # FIXME: we currently do not handle failed tuning evals
    des.list = split(design, rownames(design))
    ys = parallelMap(function(x) {
        learner = setHyperPars(learner, par.vals = as.list(x))
        model = llama.fun(learner, ldf)
        score = mean(metric(ldf, model))

        if(!quiet) message(paste("      [", paste(colnames(x), x, sep = " = ", collapse = ", "), "], score = ", score, sep = ""))
        return(score)
    }, des.list, simplify = TRUE)
    if(attr(metric, "minimize")) {
        best.i = getMinIndex(ys)
    } else {
        best.i = getMaxIndex(ys)
    }
    best.parvals = des.list[[best.i]]
    if(!quiet) message(paste("Best: [", paste(colnames(best.parvals), best.parvals, sep = " = ", collapse = ", "), "], score = ", ys[best.i], sep = ""))

    return(best.parvals)
}
