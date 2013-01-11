cptk.regression <-
function(regressor=NULL, data=NULL, split=NULL, pre=NULL, eval=NULL, combine=min) {
    splitdata = if(is.function(split)) { split(data) } else { split }
    if(class(eval) != "list") { eval = list(eval) }
    scores = vector(mode="list", length=length(eval))

    for(i in 1:length(splitdata$train)) {
        performancePredictions = matrix(nrow=nrow(splitdata$test[[i]]), ncol=length(data$performance))

        trf = list(features=subset(splitdata$train[[i]], T, data$features))
        if(!is.null(pre)) {
            trf = pre(trf$features)
        }
        tsf = list(features=subset(splitdata$test[[i]], T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        for (id in 1:length(data$performance)) {
            model = regressor(splitdata$train[[i]][[data$performance[id]]]~., data=trf$features)
            performancePredictions[,id] = predict(model, tsf$features)
        }

        predictions = apply(performancePredictions, 1, function(x) { head(which(x %in% combine(x)), 1) })

        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id,], T, data$performance)
            successes = subset(splitdata$test[[i]][id,], T, data$success)
            for (j in 1:length(eval)) {
                scores[[j]] = append(scores[[j]], as.numeric(eval[[j]](perfs, perfs[predictions[id]], successes[predictions[id]])))
            }
        }
    }

    fs = list(features=subset(data$data, T, data$features))
    if(!is.null(pre)) {
        fs = pre(fs$features)
    }
    models = lapply(1:length(data$performance), function(id) {
        return(regressor(data$data[[data$performance[id]]]~., data=fs$features))
    })
    return(list(eval=scores, predictor=function(x) {
        tsf = list(features=subset(x, T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }
        performancePredictions = matrix(nrow=nrow(x), ncol=length(data$performance))
        for (id in 1:length(data$performance)) {
            performancePredictions[,id] = predict(models[[id]], tsf$features)
        }
        return(apply(performancePredictions, 1, function(y) { factor(data$performance[head(which(y %in% combine(y)), 1)]) }))
    }))
}
