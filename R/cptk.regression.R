cptk.regression <-
function(regressor=NULL, data=NULL, pre=NULL, combine=min) {
    if(length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }
    predictions = list()

    for(i in 1:length(data$train)) {
        performancePredictions = matrix(nrow=nrow(data$test[[i]]), ncol=length(data$performance))

        trf = list(features=subset(data$train[[i]], T, data$features))
        if(!is.null(pre)) {
            trf = pre(trf$features)
        }
        tsf = list(features=subset(data$test[[i]], T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        for (id in 1:length(data$performance)) {
            model = regressor(data$train[[i]][[data$performance[id]]]~., data=trf$features)
            performancePredictions[,id] = predict(model, tsf$features)
        }

        predictions = append(predictions, list(apply(performancePredictions, 1, function(x) { data$performance[head(which(x %in% combine(x)), 1)] })))
    }

    fs = list(features=subset(data$data, T, data$features))
    if(!is.null(pre)) {
        fs = pre(fs$features)
    }
    models = lapply(1:length(data$performance), function(id) {
        return(regressor(data$data[[data$performance[id]]]~., data=fs$features))
    })
    return(list(predictions=predictions, predictor=function(x) {
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
