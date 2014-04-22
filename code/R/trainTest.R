trainTest <-
function(data, trainpart=0.6, stratify=T) {
    tmp = ddply(data$data, if(stratify) { "best" }, function(x) {
        n = nrow(x)
        x$train = c(rep.int(1, round(n*trainpart)), rep.int(2, n-round(n*trainpart)))[sample(n, n)]
        x
    })
    parts = split(data$data, tmp$train)
    perfs = subset(parts[[1]], select=data$performance)
    order = names(sort(sapply(perfs, sum), decreasing=data$minimize))
    optfun = if(data$minimize) { which.min } else { which.max }
    sapply(parts, function(part) {
        part$best = factor(apply(subset(part, select=data$performance)[order], 1, function(x) { order[optfun(unlist(x))] }))
    })
    return(c(data, list(train = list(parts[[1]]), test = list(parts[[2]]))))
}
