cptk.cvFolds = function(data, nfolds=10) {
    require(plyr)
    tmp = ddply(data, .(best), function(x) {
        n = nrow(x)
        x$fold = rep(1:nfolds, length.out = n)[sample(n, n)]
        x
    })
    return(split(data, tmp$fold))
}

cptk.trainTest = function(data, trainpart=0.6) {
    require(plyr)
    tmp = ddply(data, .(best), function(x) {
        n = nrow(x)
        x$train = c(rep.int(1, round(n*trainpart)), rep.int(2, n-round(n*trainpart)))[sample(n, n)]
        x
    })
    parts = split(data, tmp$train)
    return(list(train=parts[[1]], test=parts[[2]]))
}
