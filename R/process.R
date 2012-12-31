cptk.cvFolds = function(data, nfolds=10) {
    require(plyr)
    tmp = ddply(data$data, .(best), function(x) {
        n = nrow(x)
        x$fold = rep(1:nfolds, length.out = n)[sample(n, n)]
        x
    })
    parts = split(data$data, tmp$fold)
    return(list(train=lapply(1:nfolds, function(x) { return(unsplit(parts[-x], 1)) }),
                test=lapply(1:nfolds, function(x) { return(parts[[x]]) })))
}

cptk.trainTest = function(data, trainpart=0.6) {
    require(plyr)
    tmp = ddply(data$data, .(best), function(x) {
        n = nrow(x)
        x$train = c(rep.int(1, round(n*trainpart)), rep.int(2, n-round(n*trainpart)))[sample(n, n)]
        x
    })
    parts = split(data$data, tmp$train)
    return(list(train=list(parts[[1]]), test=list(parts[[2]])))
}

# scale all columns to interval (-1, 1)
cptk.normalize = function(data, minValues=NULL, maxValues=NULL) {
    features = subset(data$data, T, data$features)
    if(is.null(minValues)) {
        minValues = apply(features, 2, min, na.rm=T)
    }
    if(is.null(maxValues)) {
        maxValues = apply(features, 2, max, na.rm=T)
    }
    tmp = features
    for(i in names(features)) {
        tmp[[i]] = if((maxValues[[i]] - minValues[[i]]) == 0) {
            rep.int(0, length(features[[i]]))
        } else {
            ((features[[i]] - minValues[[i]]) / (maxValues[[i]] - minValues[[i]]) * 2) - 1
        }
    }

    return(list(data=cbind(tmp, subset(data$data, T, setdiff(names(data$data), data$features))), features=data$features, performance=data$performance, success=data$success, minValues=minValues, maxValues=maxValues))
}
