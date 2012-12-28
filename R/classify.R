cptk.classify = function(classifier=NULL, data=NULL, split=NULL, eval=NULL) {
    splitdata = split(data)
    model = classifier(splitdata$train$best~., subset(splitdata$train, T, data$features))
    predictions = predict(model, subset(splitdata$test, T, data$features))

    score = 0
    for (id in c(1:nrow(splitdata$test))) {
        perfs = subset(splitdata$test[id,], T, data$performance)
        score = score + as.numeric(eval(perfs, perfs[which(data$performance==predictions[id])]))
    }
    return(score)
}
