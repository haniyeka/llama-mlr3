cptk.regression <-
function(regressor=NULL, data=NULL, pre=NULL, combine=min, stack=F, expand=identity) {
    if(length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }
    predictions = list()

    for(i in 1:length(data$train)) {
        trf = list(features=subset(data$train[[i]], T, data$features))
        if(!is.null(pre)) {
            trf = pre(trf$features)
        }
        tsf = list(features=subset(data$test[[i]], T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }

        trainpredictions = matrix(nrow=nrow(trf$features), ncol=length(data$performance))
        performancePredictions = matrix(nrow=nrow(tsf$features), ncol=length(data$performance))
        for (j in 1:length(data$performance)) {
            model = regressor(data$train[[i]][[data$performance[j]]]~., data=trf$features)
            if(stack) {
                trainpredictions[,j] = predict(model, trf$features)
            }
            performancePredictions[,j] = predict(model, tsf$features)
        }

        if(stack) {
            combinedmodel = combine(data$train[[i]]$best~., data=data.frame(expand(trainpredictions)))
            combinedpredictions = as.character(predict(combinedmodel, data.frame(expand(performancePredictions))))
        } else {
            combinedpredictions = apply(performancePredictions, 1, function(x) { data$performance[head(which(x %in% combine(x)), 1)] })
        }
        predictions = append(predictions, list(combinedpredictions))

        fs = list(features=subset(data$data, T, data$features))
        if(!is.null(pre)) {
            fs = pre(fs$features)
        }
        models = lapply(1:length(data$performance), function(i) {
            return(regressor(data$data[[data$performance[i]]]~., data=fs$features))
        })
        if(stack) {
            trainpredictions = matrix(nrow=nrow(fs$features), ncol=length(data$performance))
            for (i in 1:length(data$performance)) {
                trainpredictions[,i] = predict(models[[i]], fs$features)
            }
            combinedmodel = combine(data$data$best~., data=data.frame(expand(trainpredictions)))
        }
    }

    return(list(predictions=predictions, predictor=function(x) {
        tsf = list(features=subset(x, T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }

        performancePredictions = matrix(nrow=nrow(tsf$features), ncol=length(data$performance))
        for (i in 1:length(data$performance)) {
            performancePredictions[,i] = predict(models[[i]], tsf$features)
        }

        if(stack) {
            combinedpredictions = as.character(predict(combinedmodel, data.frame(expand(performancePredictions))))
        } else {
            combinedpredictions = apply(performancePredictions, 1, function(x) { data$performance[head(which(x %in% combine(x)), 1)] })
        }
        return(combinedpredictions)
    }))
}
