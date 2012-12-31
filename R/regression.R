cptk.regression = function(regressor=NULL, data=NULL, split=NULL, pre=NULL, eval=NULL, combine=min) {
    splitdata = split(data)
    if(class(eval) != "list") { eval = list(eval) }
    scores = rep.int(0, length(eval))

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
            model = regressor(splitdata$train[[i]][[data$performance[id]]]~., trf$features)
            performancePredictions[,id] = predict(model, tsf$features)
        }

        predictions = apply(performancePredictions, 1, function(x) { head(which(x %in% combine(x)), 1) })

        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id,], T, data$performance)
            successes = subset(splitdata$test[[i]][id,], T, data$success)
            for (j in 1:length(eval)) {
                scores[j] = scores[j] + as.numeric(eval[[j]](perfs, perfs[predictions[id]], successes[predictions[id]]))
            }
        }
    }
    return(scores)
}
