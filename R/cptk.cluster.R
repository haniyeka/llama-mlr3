cptk.cluster <-
function(clusterer=NULL, data=NULL, pre=NULL) {
    if(length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }
    if(!is.list(clusterer)) { clusterer = list(clusterer) }
    combinator = "majority"
    if(!is.null(clusterer$.combine)) {
        combinator = clusterer$.combine
        clusterer = clusterer[-which(names(clusterer) == ".combine")]
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

        trainpredictions = matrix(nrow=nrow(trf$features), ncol=length(clusterer))
        ensemblepredictions = matrix(nrow=nrow(tsf$features), ncol=length(clusterer))
        for(j in 1:length(clusterer)) {
            model = clusterer[[j]](trf$features)
            trainclusters = predict(model)
            best = by(data$train[[i]], trainclusters, function(x) { names(sort(table(x$best), decreasing=T))[1] })
            if(is.function(combinator)) { # only do this if we need it
                trainpredictions[,j] = sapply(predict(model, trf$features),
                function(x) {
                    as.character(if(is.na(x)) {
                        names(sort(table(data$train[[i]]$best), decreasing=T))[1]
                    } else {
                        best[[which(names(best)==x)]]
                    })
                })
            }
            ensemblepredictions[,j] = sapply(predict(model, tsf$features),
                function(x) {
                    as.character(if(is.na(x)) {
                        names(sort(table(data$train[[i]]$best), decreasing=T))[1]
                    } else {
                        best[[which(names(best)==x)]]
                    })
                })
        }
        if(is.function(combinator)) {
            combinedmodel = combinator(data$train[[i]]$best~., data=data.frame(trainpredictions))
            combinedpredictions = as.character(predict(combinedmodel, data.frame(ensemblepredictions)))
        } else {
            combinedpredictions = apply(ensemblepredictions, 1, function(l) { l[which(table(l) %in% max(table(l)))[1]] })
        }
        predictions = append(predictions, list(combinedpredictions))
    }

    return(list(predictions=predictions, predictor=function(x) {
        fs = list(features=subset(data$data, T, data$features))
        if(!is.null(pre)) {
            fs = pre(fs$features)
        }
        tsf = list(features=subset(x, T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }

        ensemblepredictions = matrix(nrow=nrow(tsf$features), ncol=length(clusterer))
        trainpredictions = matrix(nrow=nrow(fs$features), ncol=length(clusterer))
        for(j in 1:length(clusterer)) {
            model = clusterer[[j]](fs$features)
            clusters = predict(model)
            best = by(data$data, clusters, function(x) { names(sort(table(x$best), decreasing=T))[1] })
            if(is.function(combinator)) {
                trainpredictions[,j] = sapply(predict(model, fs$features),
                    function(x) {
                        as.character(if(is.na(x)) {
                            names(sort(table(data$data$best), decreasing=T))[1]
                        } else {
                            best[[which(names(best)==x)]]
                        })
                    })
            }
            ensemblepredictions[,j] = sapply(predict(model, tsf$features),
                function(x) {
                    as.character(if(is.na(x)) {
                        names(sort(table(data$data$best), decreasing=T))[1]
                    } else {
                        best[[which(names(best)==x)]]
                    })
                })
        }
        if(is.function(combinator)) {
            combinedmodel = combinator(data$data$best~., data=data.frame(trainpredictions))
            combinedpredictions = as.character(predict(combinedmodel, data.frame(ensemblepredictions)))
        } else {
            combinedpredictions = apply(ensemblepredictions, 1, function(l) { l[which(table(l) %in% max(table(l)))[1]] })
        }
        return(combinedpredictions)
    }))
}
