cptk.read <-
function(featurefilename, perffilename, successfilename=NULL, minimize=T) {
    ff = read.csv(featurefilename)
    pf = read.csv(perffilename)
    common = intersect(names(ff), names(pf))
    pnames = setdiff(names(pf), common)
    combined = merge(ff, pf)
    optfun = if(minimize) { min } else { max }
    combined$best = factor(apply(combined, 1, function(x) { pnames[head(which(unlist(x[pnames]) == optfun(unlist(x[pnames]))), 1)] }))

    snames = NULL
    if(!is.null(successfilename)) {
        sf = read.csv(successfilename)
        names(sf) = c(head(names(sf), 1), sapply(tail(names(sf), -1), function(x) { paste(x, "success", sep="_") }))
        combined = merge(combined, sf)
        snames = setdiff(names(sf), common)
    }

    return(list(data=combined, features=setdiff(names(ff), common), performance=pnames, success=snames))
}
