cptk.read = function(featurefilename, perffilename, minimize=T) {
    ff = read.csv(featurefilename)
    pf = read.csv(perffilename)
    common = intersect(names(ff), names(pf))
    pnames = setdiff(names(pf), common)
    combined = merge(ff, pf)
    optfun = if(minimize) { min } else { max }
    combined$best = factor(apply(combined, 1, function(x) { pnames[head(which(unlist(x[pnames]) == optfun(unlist(x[pnames]))), 1)] }))
    return(list(data=combined, features=setdiff(names(ff), common), performance=pnames))
}
