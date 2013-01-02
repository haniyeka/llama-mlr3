cptk.trainTest <-
function(data, trainpart=0.6) {
    require(plyr)
    tmp = ddply(data$data, .(best), function(x) {
        n = nrow(x)
        x$train = c(rep.int(1, round(n*trainpart)), rep.int(2, n-round(n*trainpart)))[sample(n, n)]
        x
    })
    parts = split(data$data, tmp$train)
    return(list(train=list(parts[[1]]), test=list(parts[[2]])))
}
