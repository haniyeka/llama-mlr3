input <-
function(features, performances, successes=NULL, costs=NULL, minimize=T) {
    if(nrow(features) != nrow(performances) || (!is.null(successes) && (nrow(features) != nrow(successes) || nrow(performances) != nrow(successes))) || (!is.null(costs) && (nrow(features) != nrow(costs) || nrow(performances) != nrow(costs)))) {
        warning("Different number of rows in data frames, taking only common rows.")
    }
    common = intersect(names(features), names(performances))
    if(length(common) == 0) {
        stop("Performance can't be linked to features -- no common columns!")
    }
    pnames = setdiff(names(performances), common)
    combined = merge(features, performances, by=common)
    optfun = if(minimize) { min } else { max }
    combined$best = factor(apply(subset(combined, T, pnames), 1, function(x) { pnames[head(which(unlist(x) == optfun(unlist(x))), 1)] }))

    snames = NULL
    if(!is.null(successes)) {
        commonSuccess = intersect(names(successes), common)
        successNames = setdiff(names(successes), common)
        if(length(commonSuccess) == 0) {
            stop("Successes can't be linked to rest of data -- no common columns!")
        }
        if(length(intersect(successNames, pnames)) != length(successNames)) {
            stop(paste("Successes (", paste(successNames, collapse=", "), ") and performances (", paste(pnames, collapse=", "), ") for different algorithms given!", sep=""))
        }
        names(successes) = c(commonSuccess, sapply(successNames, function(x) { paste(x, "success", sep="_") }))
        combined = merge(combined, successes, by=common)
        snames = setdiff(names(successes), common)
    }

    retval = list(data=combined, features=setdiff(names(features), common), performance=pnames, success=snames)


    if(!is.null(costs)) {
        if(is.numeric(costs)) {
            # same cost for everything
            combined$cost = rep.int(costs, times=length(combined$best))
            cnames = c("cost")
        } else {
            commonCosts = intersect(names(costs), common)
            if(length(commonCosts) == 0) {
                stop("Costs can't be linked to rest of data -- no common columns!")
            }
            combined = merge(combined, costs, by=common)
            cnames = setdiff(names(costs), common)
        }
        retval$cost = cnames
    }

    return(retval)
}
