cptk.classify = function(classifier, train, test, features, performance, evalFunction) {
    model = classifier(train$best~., subset(train, T, features))
    predictions = predict(model, subset(test, T, features))

    score = 0
    for (id in c(1:nrow(test))) {
        perfs = subset(test[id,], T, performance)
        score = score + evalFunction(perfs, perfs[which(performance==predictions[id])])
    }
    return(score)
}
