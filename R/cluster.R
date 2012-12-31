cptk.cluster = function(clusterer=NULL, data=NULL, split=NULL, eval=NULL) {
    splitdata = split(data)
    score = 0

    for(i in 1:length(splitdata$train)) {
        model = clusterer(subset(splitdata$train[[i]], T, data$features))
        trainclusters = predict(model)
        best = by(splitdata$train[[i]], trainclusters, function(x) { names(sort(table(x$best), decreasing=T))[1] })
        testclusters = predict(model, subset(splitdata$test[[i]], T, data$features))

        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id,], T, data$performance)
            successes = subset(splitdata$test[[i]][id,], T, data$success)
            score = score + as.numeric(eval(perfs, perfs[which(data$performance==best[testclusters[id]+1])], successes[which(data$performance==best[testclusters[id]+1])]))
        }
    }
    return(score)
}
