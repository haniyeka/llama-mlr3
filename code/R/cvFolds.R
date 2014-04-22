cvFolds <-
function(data, nfolds=10, stratify=T) {
    if(nfolds == -1) {
        stratify = F
        nfolds = nrow(data$data)
    }
    tmp = ddply(data$data, if(stratify) { "best" }, function(x) {
        n = nrow(x)
        x$fold = rep(1:nfolds, length.out = n)[sample(n, n)]
        x
    })
    parts = split(data$data, tmp$fold)
    return(c(data,
            list(train = lapply(1:nfolds, function(x) {
                    part = unsplit(parts[-x], tmp$fold[tmp$fold!=x])
                    perfs = subset(part, select=data$performance)
                    order = names(sort(sapply(perfs, sum), decreasing=!data$minimize))
                    optfun = if(data$minimize) { which.min } else { which.max }
                    part$best = factor(apply(perfs[order], 1, function(x) { order[optfun(unlist(x))] }))
                    return(part)
                }),
             test = lapply(1:nfolds, function(x) {
                    part = parts[[x]]
                    otherPart = unsplit(parts[-x], tmp$fold[tmp$fold!=x])
                    perfs = subset(part, select=data$performance)
                    order = names(sort(sapply(subset(otherPart, select=data$performance), sum), decreasing=!data$minimize))
                    optfun = if(data$minimize) { which.min } else { which.max }
                    part$best = factor(apply(perfs[order], 1, function(x) { order[optfun(unlist(x))] }))
                    return(part)
                 }))))
}
