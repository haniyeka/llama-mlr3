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
    retval = do.call(rbind, lapply(1:n.outer.folds, function(i) {
        if(!quiet) message(paste("Fold ", i, "/", n.outer.folds, ":", sep = ""))
        ldf2 = ldf
        ldf2$data = ldf$data[ldf$train[[i]],]
        ldf3 = ldf2
        ldf3 = cvFolds(ldf3, nfolds = nfolds)
        inner.res = tuneLlamaModel(ldf3, llama.fun, learner, design, metric, quiet)

        outer.split.ldf = ldf
        outer.split.ldf$train = list(ldf$train[[i]])
        outer.split.ldf$test = list(ldf$test[[i]])
        sbscore = mean(metric(outer.split.ldf, singleBest))
        ys = parallelMap(function(x) {
            pars = as.list(design[x,,drop = FALSE])
            learner2 = setHyperPars(learner, par.vals = pars)
            model = llama.fun(learner2, data = outer.split.ldf)
            sbscore = mean(metric(ldf, singleBest))
            score = mean(metric(outer.split.ldf, model))/sbscore

            if(!quiet) message(paste("outer: [", paste(names(pars), pars, sep = " = ", collapse = ", "), "], score = ", score, sep = ""))
            return(score)
        }, 1:nrow(design), simplify = TRUE, level = "llama.tune")
        outer.res = data.frame(outer.score = ys)

        res = cbind(design, inner.res, outer.res)
        res$fold = i
        return(res)
    }))

    return(retval)
}

tuneLlamaModel <-
function(ldf, llama.fun, learner, design, metric, quiet) {
    # FIXME: we currently do not handle failed tuning evals
    ys = parallelMap(function(x) {
        pars = as.list(design[x,,drop = FALSE])
        learner = setHyperPars(learner, par.vals = pars)
        model = llama.fun(learner, ldf)
        sbscore = mean(metric(ldf, singleBest))
        score = mean(metric(ldf, model))/sbscore

        if(!quiet) message(paste("inner: [", paste(names(pars), pars, sep = " = ", collapse = ", "), "], score = ", score, sep = ""))
        return(score)
    }, 1:nrow(design), simplify = TRUE, level = "llama.tune")

    return(data.frame(inner.score = ys))
}
