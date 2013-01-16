cptk.classify <-
function(classifier=NULL, data=NULL, pre=NULL) {
    if(length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }
    if(!is.list(classifier)) { classifier = list(classifier) }
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

        ensemblepredictions = matrix(nrow=nrow(tsf$features), ncol=length(classifier))
        for(j in 1:length(classifier)) {
            model = classifier[[j]](data$train[[i]]$best~., data=trf$features)
            ensemblepredictions[,j] = predict(model, tsf$features)
        }
        predictions = append(predictions, list(sapply(apply(ensemblepredictions, 1, function(l) { l[which(table(l) %in% max(table(l)))[1]] }), function(x) { data$performance[x] })))
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
        for(i in 1:length(classifier)) {
            model = classifier[[i]](data$data$best~., data=fs$features)
            ensemblepredictions[,i] = predict(model, tsf$features)
        }
        predictions = sapply(apply(ensemblepredictions, 1, function(l) { l[which(table(l) %in% max(table(l)))[1]] }), function(x) { data$performance[x] })
        return(predictions)
    }))
}
