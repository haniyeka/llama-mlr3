cptk.classify = function(classifier=NULL, data=NULL, split=NULL, eval=NULL) {
    splitdata = split(data)
    score = 0

    for(i in 1:length(splitdata$train)) {
        model = classifier(splitdata$train[[i]]$best~., subset(splitdata$train[[i]], T, data$features))
        predictions = predict(model, subset(splitdata$test[[i]], T, data$features))

        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id,], T, data$performance)
            successes = subset(splitdata$test[[i]][id,], T, data$success)
            score = score + as.numeric(eval(perfs, perfs[which(data$performance==predictions[id])], successes[which(data$performance==predictions[id])]))
        }
    }
    return(score)
}
