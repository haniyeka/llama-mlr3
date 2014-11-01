cvFolds <-
function(data, nfolds=10, stratify=T) {
    if(nfolds == -1) {
        stratify = F
        nfolds = nrow(data$data)
    }
    if(stratify) {
        data$data$stratifier = sapply(data$data$best, paste, collapse="-")
    }
    tmp = ddply(data$data, if(stratify) "stratifier", function(x) {
        n = nrow(x)
        x$id = rep(1:nfolds, length.out = n)[sample(n, n)]
        x
    })
    data$data$stratifier = NULL
    if(length(unique(tmp$id)) != nfolds) {
        stop(paste("Requested ", nfolds, " folds, but cannot produce this many.", sep=""))
    }
    parts = split(1:nrow(data$data), tmp$id)

    newdata = c(data,
            list(train = lapply(1:nfolds, function(x) { return(unsplit(parts[-x], tmp$id[tmp$id!=x])) }),
                 test = lapply(1:nfolds, function(x) { return(parts[[x]]) })))
    class(newdata) = "llama.data"
    attr(newdata, "hasSplits") = TRUE
    return(newdata)
}
