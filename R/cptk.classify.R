cptk.classify <-
function(classifier=NULL, data=NULL, pre=NULL, eval=NULL) {
    if(length(data$train) == 0 || length(data$test) == 0) {
        stop("Need data with train/test split!")
    }
    if(class(eval) != "list") { eval = list(eval) }
    scores = vector(mode="list", length=length(eval))

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
        predictions = predict(model, tsf$features)

        for (id in 1:nrow(data$test[[i]])) {
            perfs = subset(data$test[[i]][id,], T, data$performance)
            successes = subset(data$test[[i]][id,], T, data$success)
            for (j in 1:length(eval)) {
                scores[[j]] = append(scores[[j]], as.numeric(eval[[j]](perfs, perfs[which(data$performance==predictions[id])], successes[which(data$performance==predictions[id])])))
            }
        }
    }

    fs = list(features=subset(data$data, T, data$features))
    if(!is.null(pre)) {
        fs = pre(fs$features)
    }
    model = classifier(data$data$best~., data=fs$features)
    return(list(eval=scores, predictor=function(x) {
        tsf = list(features=subset(x, T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, fs$meta)
        }
        return(predict(model, tsf$features))
    }))
}
