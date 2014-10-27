predNames = c("algorithm", "score")
vbs <-
function(data=NULL) {
    if(is.null(data)) {
        stop("Need data to determine virtual best!")
    }
    return(lapply(data$data$best, function(l) {
        if(length(l) == 1 && is.na(l)) {
            setNames(data.frame(foo = NA, bar = 0), predNames)
        } else {
            setNames(data.frame(as.table(sort(table(l), decreasing=T))), predNames)
        }
    }))
}
class(vbs) = "llama.model"
attr(vbs, "hasPredictions") = FALSE

singleBestByCount <-
function(data=NULL) {
    if(is.null(data)) {
        stop("Need data to determine single best!")
    }
    best = breakBestTies(data)
    preds = rep.int(names(sort(table(best), decreasing=T)[1]), length(data$data$best))
    return(lapply(preds, function(l) { setNames(data.frame(table(l)), predNames) }))
}
class(singleBestByCount) = "llama.model"
attr(singleBestByCount, "hasPredictions") = FALSE

singleBestByPar <-
function(data=NULL, factor=10) {
    if(is.null(data)) {
        stop("Need data to determine single best!")
    }
    if(is.null(data$success)) {
        stop("Need successes to compute PAR scores.")
    }

    best = names(sort(sapply(data$performance, function(x) {
        prs = replicate(length(data$data$best), setNames(data.frame(table(x)), predNames), simplify=F)
        model = list(predictions=prs)
        class(model) = "llama.model"
        attr(model, "hasPredictions") = TRUE
        sum(parscores(data, model, factor=factor))
    })))[1]
    preds = rep.int(best, length(data$data$best))
    return(lapply(preds, function(l) { setNames(data.frame(table(l)), predNames) }))
}
class(singleBestByPar) = "llama.model"
attr(singleBestByPar, "hasPredictions") = FALSE

singleBestBySuccesses <-
function(data=NULL) {
    if(is.null(data)) {
        stop("Need data to determine single best!")
    }
    if(is.null(data$success)) {
        stop("Need successes to compute successes.")
    }

    best = names(sort(sapply(data$performance, function(x) {
        model = list(predictions=replicate(length(data$data$best), setNames(data.frame(table(x)), predNames), simplify=F))
        class(model) = "llama.model"
        attr(model, "hasPredictions") = TRUE
        sum(successes(data, model))
    }), decreasing=T))[1]
    preds = rep.int(best, length(data$data$best))
    return(lapply(preds, function(l) { setNames(data.frame(table(l)), predNames) }))
}
class(singleBestBySuccesses) = "llama.model"
attr(singleBestBySuccesses, "hasPredictions") = FALSE

singleBest <-
function(data=NULL) {
    if(is.null(data)) {
        stop("Need data to determine single best!")
    }

    best = names(sort(sapply(data$performance, function(x) { sum(data$data[x]) }), decreasing=!data$minimize))[1]
    preds = rep.int(best, length(data$data$best))
    return(lapply(preds, function(l) { setNames(data.frame(table(l)), predNames) }))
}
class(singleBest) = "llama.model"
attr(singleBest, "hasPredictions") = FALSE

breakBestTies <-
function(data=NULL, fold=NULL) {
    if(is.null(data)) {
        stop("Need data to break ties!")
    }

    if(is.null(fold)) {
        perfs = data$data[data$performance]
    } else {
        perfs = data$train[[fold]][data$performance]
    }
    order = names(sort(sapply(perfs, sum), decreasing=!data$minimize))
    optfun = if(data$minimize) { which.min } else { which.max }
    best = factor(apply(perfs[order], 1, function(x) { order[optfun(unlist(x))] }))
    names(best) = NULL
    return(best)
}
