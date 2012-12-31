cptk.cluster = function(clusterer=NULL, data=NULL, split=NULL, pre=NULL, eval=NULL) {
    splitdata = split(data)
    if(class(eval) != "list") { eval = list(eval) }
    scores = rep.int(0, length(eval))

    for(i in 1:length(splitdata$train)) {
        trf = list(features=subset(splitdata$train[[i]], T, data$features))
        if(!is.null(pre)) {
            trf = pre(trf$features)
        }
        model = clusterer(trf$features)
        trainclusters = predict(model)
        best = by(splitdata$train[[i]], trainclusters, function(x) { names(sort(table(x$best), decreasing=T))[1] })
        tsf = list(features=subset(splitdata$test[[i]], T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        testclusters = predict(model, tsf$features)

        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id,], T, data$performance)
            successes = subset(splitdata$test[[i]][id,], T, data$success)
            for (j in 1:length(eval)) {
                scores[j] = scores[j] + as.numeric(eval[[j]](perfs, perfs[which(data$performance==best[[which(names(best)==testclusters[id])]])], successes[which(data$performance==best[[which(names(best)==testclusters[id])]])]))
            }
        }
    }

    fs = list(features=subset(data$data, T, data$features))
    if(!is.null(pre)) {
        fs = pre(fs$features)
    }
    model = clusterer(fs$features)
    clusters = predict(model)
    best = by(data$data, clusters, function(x) { names(sort(table(x$best), decreasing=T))[1] })
    return(list(eval=scores, predictor=function(x) {
        tsf = list(features=subset(x, T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }
        return(sapply(predict(model, tsf$features), function(y) {
            return(factor(best[[which(names(best)==y)]]))
        }))
    }))
}
