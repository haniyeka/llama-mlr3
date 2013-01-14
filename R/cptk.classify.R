cptk.classify <-
function(classifier=NULL, data=NULL, pre=NULL) {
    if(length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }
    predictions = list()

    for(i in 1:length(data$train)) {
        trf = list(features=subset(data$train[[i]], T, data$features))
        if(!is.null(pre)) {
            trf = pre(trf$features)
        }
        model = classifier(data$train[[i]]$best~., data=trf$features)
        tsf = list(features=subset(data$test[[i]], T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        predictions = append(predictions, list(sapply(predict(model, tsf$features), function(x) { data$performance[x] })))
    }

    fs = list(features=subset(data$data, T, data$features))
    if(!is.null(pre)) {
        fs = pre(fs$features)
    }
    model = classifier(data$data$best~., data=fs$features)
    return(list(predictions=predictions, predictor=function(x) {
        tsf = list(features=subset(x, T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }
        return(sapply(predict(model, tsf$features), function(x) { data$performance[x] }))
    }))
}
