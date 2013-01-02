cptk.cvFolds <-
function(data, nfolds=10) {
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
