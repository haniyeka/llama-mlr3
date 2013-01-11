cptk.classify <-
function(classifier=NULL, data=NULL, split=NULL, pre=NULL, eval=NULL) {
    splitdata = if(is.function(split)) { split(data) } else { split }
    if(class(eval) != "list") { eval = list(eval) }
    scores = rep.int(0, length(eval))

    for(i in 1:length(splitdata$train)) {
        trf = list(features=subset(splitdata$train[[i]], T, data$features))
        if(!is.null(pre)) {
            trf = pre(trf$features)
        }
        model = classifier(splitdata$train[[i]]$best~., data=trf$features)
        tsf = list(features=subset(splitdata$test[[i]], T, data$features))
        if(!is.null(pre)) {
            tsf = pre(tsf$features, trf$meta)
        }
        predictions = predict(model, tsf$features)

        for (id in 1:nrow(splitdata$test[[i]])) {
            perfs = subset(splitdata$test[[i]][id,], T, data$performance)
            successes = subset(splitdata$test[[i]][id,], T, data$success)
            for (j in 1:length(eval)) {
                scores[j] = scores[j] + as.numeric(eval[[j]](perfs, perfs[which(data$performance==predictions[id])], successes[which(data$performance==predictions[id])]))
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
