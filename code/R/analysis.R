contributions <-
function(data=NULL, minimize=TRUE, limit=100) {
    if(is.null(data)) {
        stop("Need data to determine contributions!")
    }
    subsets = unlist(lapply(1:(length(data$performance)-1), function(l) {
        combn(data$performance, l, simplify=F)
    }), recursive=F)
    optfun = if(minimize) { min } else { max }
    contributions = sapply(data$performance, function(alg) {
        withoutAlg = Filter(function(s) { (!(alg %in% s)) }, subsets)
        if(length(withoutAlg) > limit) {
            withoutAlg = sample(withoutAlg, limit)
        }
        shapelys = sapply(withoutAlg, function(s) {
            perfWo = sum(apply(subset(data$data, T, s), 1, function(x) { optfun(unlist(x)) }))
            perfW = sum(apply(subset(data$data, T, c(s, alg)), 1, function(x) { optfun(unlist(x)) }))
            if(minimize) {
                perfWo - perfW
            } else {
                perfW - perfWo
            }
        })
        mean(shapelys)
    })
    return(contributions)
}
