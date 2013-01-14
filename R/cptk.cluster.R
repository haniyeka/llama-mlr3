cptk.cluster <-
function(clusterer=NULL, data=NULL, pre=NULL) {
    if(length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }
    predictions = list()

    for(i in 1:length(data$train)) {
        trf = list(features=subset(data$train[[i]], T, data$features))
        if(!is.null(pre)) {
            trf = pre(trf$features)
        }
        model = clusterer(trf$features)
        trainclusters = predict(model)
        best = by(data$train[[i]], trainclusters, function(x) { names(sort(table(x$best), decreasing=T))[1] })
        tsf = list(features=subset(data$test[[i]], T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        testclusters = predict(model, tsf$features)
        predictions = append(predictions, list(sapply(predict(model, tsf$features),
            function(x) {
                if(is.na(x)) {
                    names(sort(table(data$train[[i]]$best), decreasing=T))[1]
                } else {
                    best[[which(names(best)==x)]]
                }
            })))
    }

    fs = list(features=subset(data$data, T, data$features))
    if(!is.null(pre)) {
        fs = pre(fs$features)
    }
    model = clusterer(fs$features)
    clusters = predict(model)
    best = by(data$data, clusters, function(x) { names(sort(table(x$best), decreasing=T))[1] })
    return(list(predictions=predictions, predictor=function(x) {
        tsf = list(features=subset(x, T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }
        return(sapply(predict(model, tsf$features), function(y) {
            if(is.na(y)) {
                return(factor(names(sort(table(data$data$best), decreasing=T))[1]))
            } else {
                return(factor(best[[which(names(best)==y)]]))
            }
        }))
    }))
}
