cptk.input <-
function(ff, pf, sf=NULL, minimize=T) {
    if(nrow(ff) != nrow(pf) || (!is.null(sf) && (nrow(ff) != nrow(sf) || nrow(pf) != nrow(sf)))) {
        warning("Different number of rows in data frames, taking only common rows.")
    }
    common = intersect(names(ff), names(pf))
    pnames = setdiff(names(pf), common)
    combined = merge(ff, pf)
    optfun = if(minimize) { min } else { max }
    combined$best = factor(apply(combined, 1, function(x) { pnames[head(which(unlist(x[pnames]) == optfun(unlist(x[pnames]))), 1)] }))

    snames = NULL
    if(!is.null(sf)) {
        names(sf) = c(intersect(names(sf), common), sapply(setdiff(names(sf), common), function(x) { paste(x, "success", sep="_") }))
        combined = merge(combined, sf)
        snames = setdiff(names(sf), common)
    }

    return(list(data=combined, features=setdiff(names(ff), common), performance=pnames, success=snames))
}
