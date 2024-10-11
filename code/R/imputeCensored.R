imputeCensored <-
function(data=NULL, estimator=lrn("regr.lm"), epsilon=0.1, maxit=1000) {
    if(!testClass(estimator, "Learner")) {
        stop("Need regressor to impute values!")
    }
    assertClass(data, "llama.data")
    if(is.null(data$success)) {
        stop("Need successes to impute censored values!")
    }
    if(epsilon <= 0) {
        stop("Epsilon must be > 0!")
    }

    data$original_data = data$data

    i = 0
    for(i in 1:length(data$success)) {
        s = data$success[i]
        p = data$performance[i]
        if(!any(data$data[[s]])) {
            stop(paste("Cannot impute for ", p, ", no non-censored values!"), sep="")
        }
        if(!all(data$data[[s]])) {
            splits = split(1:nrow(data$data), data$data[[s]])
            haveind = splits$`TRUE`
            wantind = splits$`FALSE`
            if(is.null(data$algorithmFeatures)) {
                task = TaskRegr$new(id="imputation", target="target", backend=cbind(data.frame(target=data$data[haveind,p]), data$data[haveind,][data$features]))
                model = estimator$train(task=task)
                data$data[wantind,p] = model$predict_newdata(newdata=data$data[wantind,][data$features])$response
            } else {
                task = TaskRegr$new(id="imputation", target="target", backend=cbind(data.frame(target=data$data[haveind,p]), data$data[haveind,][c(data$features, data$algorithmFeatures)]))
                model = estimator$train(task=task)
                data$data[wantind,p] = model$predict_newdata(newdata=data$data[wantind,][c(data$features, data$algorithmFeatures)])$response
            }

            diff = Inf
            it = 1
            while(diff > epsilon) {
                if(is.null(data$algorithmFeatures)) {
                    task = TaskRegr$new(id="imputation", target="target", backend=cbind(data.frame(target=data$data[[p]]), data$data[data$features]))
                    model = estimator$train(task=task)
                    preds = model$predict_newdata(newdata=data$data[wantind,][data$features])$response
                } else {
                    task = TaskRegr$new(id="imputation", target="target", backend=cbind(data.frame(target=data$data[[p]]), data$data[c(data$features, data$algorithmFeatures)]))
                    model = estimator$train(task=task)
                    preds = model$predict_newdata(newdata=data$data[wantind,][c(data$features, data$algorithmFeatures)])$response
                }

                diff = max(abs(preds - data$data[wantind,p]))
                data$data[wantind,p] = preds
                it = it + 1
                if(it > maxit) {
                    warning(paste("Did not reach convergence within ", maxit, " iterations for ", p, ".", sep=""))
                    break
                }
            }
            data$data[[s]] = rep.int(T, nrow(data$data))
        }
    }

    return(data)
}

