cptk.classify <-
function(classifier=NULL, data=NULL, pre=NULL) {
    if(length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }
    if(!is.list(classifier)) { classifier = list(classifier) }
    combinator = "majority"
    if(!is.null(classifier$.combine)) {
        combinator = classifier$.combine
        classifier = classifier[-which(names(classifier) == ".combine")]
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

        trainpredictions = matrix(nrow=nrow(trf$features), ncol=length(classifier))
        ensemblepredictions = matrix(nrow=nrow(tsf$features), ncol=length(classifier))
        for(j in 1:length(classifier)) {
            model = classifier[[j]](data$train[[i]]$best~., data=trf$features)
            if(is.function(combinator)) { # only do this if we need it
                trainpredictions[,j] = as.character(predict(model, trf$features))
            }
            ensemblepredictions[,j] = as.character(predict(model, tsf$features))
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
        ensemblepredictions = matrix(nrow=nrow(tsf$features), ncol=length(classifier))
        trainpredictions = matrix(nrow=nrow(fs$features), ncol=length(classifier))
        for(i in 1:length(classifier)) {
            model = classifier[[i]](data$data$best~., data=fs$features)
            if(is.function(combinator)) {
                trainpredictions[,i] = as.character(predict(model, fs$features))
            }
            ensemblepredictions[,i] = as.character(predict(model, tsf$features))
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
