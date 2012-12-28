cptk.regression = function(regressor=NULL, data=NULL, split=NULL, eval=NULL, combine=min) {
    splitdata = split(data)
    score = 0

    for(i in 1:length(splitdata$train)) {
        performancePredictions = matrix(nrow=nrow(splitdata$test[[i]]), ncol=length(data$performance))

        for (id in 1:length(data$performance)) {
            model = regressor(splitdata$train[[i]][[data$performance[id]]]~., subset(splitdata$train[[i]], T, data$features))
            performancePredictions[,id] = predict(model, subset(splitdata$test[[i]], T, data$features))
        }

        predictions = apply(performancePredictions, 1, function(x) { head(which(x %in% combine(x)), 1) })

        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id,], T, data$performance)
            score = score + as.numeric(eval(perfs, perfs[predictions[id]]))
        }
    }
    return(score)
}
