regressionPairs <-
function(regressor=NULL, data=NULL, pre=function(x, y=NULL) { list(features=x) }, combine=NULL) {
    if(is.null(regressor)) {
        stop("No regressor given!")
    }
    if(is.null(data) || length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }

    totalBests = breakBestTies(data)
    combns = combn(data$performance, 2)
    predictions = parallelMap(function(i) {
        trf = pre(subset(data$train[[i]], T, data$features))
        tsf = pre(subset(data$test[[i]], T, data$features), trf$meta)

        trainpredictions = matrix(nrow=nrow(trf$features), ncol=ncol(combns))
        pairpredictions = matrix(nrow=nrow(tsf$features), ncol=ncol(combns))
        for (j in 1:ncol(combns)) {
            values = data$train[[i]][[combns[1,j]]] - data$train[[i]][[combns[2,j]]]
            model = regressor(values~., data=trf$features)
            if(is.function(combine)) { # only do this if we need it
                trainpredictions[,j] = predict(model, trf$features)
            }
            pairpredictions[,j] = predict(model, tsf$features)
        }

        if(is.function(combine)) {
            trainBests = breakBestTies(data, i)
            combinedmodel = combine(trainBests~., data=cbind(trf$features, data.frame(trainpredictions)))
            preds = as.character(predict(combinedmodel, cbind(tsf$features, data.frame(pairpredictions))))
            combinedpredictions = lapply(preds, function(l) { setNames(data.frame(table(l)), predNames) })
        } else {
            if(data$minimize) {
                sign = function(d) { return(-d); }
            } else {
                sign = function(d) { return(d); }
            }
            combinedpredictions = apply(pairpredictions, 1, function(row) {
                performanceSums = sapply(data$performance, function(p) {
                    sum(sapply(1:ncol(combns), function(k) {
                        if(p == combns[1,k]) {
                            sign(row[k])
                        } else if(p == combns[2,k]) {
                            sign(-row[k])
                        } else {
                            0
                        }
                    }))
                })
                setNames(data.frame(data$performance[sort.list(performanceSums, decreasing=T)], sort(performanceSums, decreasing=T), row.names=NULL), predNames)
            })
        }
        return(list(combinedpredictions))
    }, 1:length(data$train), simplify=T)

    fs = pre(subset(data$data, T, data$features))
    models = lapply(1:ncol(combns), function(i) {
        values = data$data[[combns[1,i]]] - data$data[[combns[2,i]]]
        return(regressor(values~., data=fs$features))
    })
    if(is.function(combine)) {
        trainpredictions = matrix(nrow=nrow(fs$features), ncol=ncol(combns))
        for(i in 1:ncol(combns)) {
            trainpredictions[,i] = predict(models[[i]], fs$features)
        }
        combinedmodel = combine(totalBests~., data=cbind(fs$features, data.frame(trainpredictions)))
    }

    return(list(predictions=predictions, models=models, predictor=function(x) {
        tsf = pre(subset(x, T, data$features), fs$meta)
        pairpredictions = matrix(nrow=nrow(tsf$features), ncol=ncol(combns))
        for(i in 1:ncol(combns)) {
            pairpredictions[,i] = predict(models[[i]], tsf$features)
        }
        if(is.function(combine)) {
            preds = as.character(predict(combinedmodel, cbind(tsf$features, data.frame(pairpredictions))))
            combinedpredictions = lapply(preds, function(l) { setNames(data.frame(table(l)), predNames) })
        } else {
            if(data$minimize) {
                sign = function(d) { return(-d); }
            } else {
                sign = function(d) { return(d); }
            }
            combinedpredictions = apply(pairpredictions, 1, function(row) {
                performanceSums = sapply(data$performance, function(p) {
                    sum(sapply(1:ncol(combns), function(k) {
                        if(p == combns[1,k]) {
                            sign(row[k])
                        } else if(p == combns[2,k]) {
                            sign(-row[k])
                        } else {
                            0
                        }
                    }))
                })
                setNames(data.frame(data$performance[sort.list(performanceSums, decreasing=T)], sort(performanceSums, decreasing=T), row.names=NULL), predNames)
            })
        }
        return(combinedpredictions)
    }))
}
