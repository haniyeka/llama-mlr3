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
cptk.normalize = function(rawfeatures, meta=NULL) {
    if(is.null(meta) || is.null(meta$minValues)) {
        meta$minValues = apply(rawfeatures, 2, min, na.rm=T)
    }
    if(is.null(meta) || is.null(meta$maxValues)) {
        meta$maxValues = apply(rawfeatures, 2, max, na.rm=T)
    }
    tmp = rawfeatures
    for(i in names(rawfeatures)) {
        tmp[[i]] = if((meta$maxValues[[i]] - meta$minValues[[i]]) == 0) {
            rep.int(0, length(rawfeatures[[i]]))
        } else {
            ((rawfeatures[[i]] - meta$minValues[[i]]) / (meta$maxValues[[i]] - meta$minValues[[i]]) * 2) - 1
        }
    }

    return(list(features=tmp, meta=meta))
}
