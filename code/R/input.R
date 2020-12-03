input <-
function(instanceFeatures, algorithmFeatures = NULL, performances, successes=NULL, costs=NULL, extra=NULL, minimize=T) {
    if(nrow(instanceFeatures) != nrow(performances) || (!is.null(successes) && (nrow(instanceFeatures) != nrow(successes) || nrow(performances) != nrow(successes)))) {
        warning("Different number of rows in data frames, taking only common rows.")
    }
    #FIXME: find algorithm column without explicitly stating it
    if(!is.null(algorithmFeatures)) {
        algos = colnames(performances)[-1] %in% algorithmFeatures$algorithm
        if(!all(algos)) {
            stop("Not all algorithms are present in algorithmic features, taking only common algorithms.")
        }
    }
    
    instanceCommon = intersect(names(instanceFeatures), names(performances))
    if(length(instanceCommon) == 0) {
        stop("Performance can't be linked to instance features -- no common columns!")
    }
    pnames = setdiff(names(performances), instanceCommon)
    
    ids = factor(do.call(paste0, instanceFeatures[instanceCommon]))
    if(length(ids) != length(levels(ids))) {
        stop("Common columns do not provide unique IDs!")
    }
    
    combined = merge(instanceFeatures, performances, by=instanceCommon)
    snames = NULL
    if(!is.null(successes)) {
        commonSuccess = intersect(names(successes), instanceCommon)
        successNames = setdiff(names(successes), instanceCommon)
        if(length(commonSuccess) == 0) {
            stop("Successes can't be linked to rest of data -- no common columns!")
        }
        if(length(intersect(successNames, pnames)) != length(successNames)) {
            stop(paste("Successes (", paste(successNames, collapse=", "), ") and performances (", paste(pnames, collapse=", "), ") for different algorithms given!", sep=""))
        }
        names(successes) = c(commonSuccess, sapply(successNames, function(x) { paste(x, "success", sep="_") }))
        # reorder successes according to performances
        successes = successes[c(commonSuccess, sapply(successNames[order(match(successNames, pnames))], function(x) { paste(x, "success", sep="_") }))]
        combined = merge(combined, successes, by=instanceCommon)
        snames = setdiff(names(successes), instanceCommon)
    }
    
    optfun = if(minimize) { min } else { max }
    best = apply(combined, 1,
                 function(x) {
                     tosel = pnames
                     if(!is.null(successes)) {
                         nosuccs = sapply(snames[which(as.logical(x[snames]) == FALSE)], function(y) { unlist(strsplit(y, "_"))[1] })
                         tosel = setdiff(pnames, nosuccs)
                     }
                     if(length(tosel) == 0) {
                         # nothing was able to solve this instance
                         return(NA)
                     } else {
                         perfs = as.numeric(x[tosel])
                         return(tosel[which(perfs == optfun(perfs))])
                     }
                 })
    
    # simplify...
    names(best) = NULL
    
    # add algorithm features
    if(!is.null(algorithmFeatures)) {
        # reshape performances into  long format
        perf = reshape(performances, direction = "long", varying = pnames, 
                       v.names = "performance", timevar = "algorithm", 
                       times = pnames, idvar = instanceCommon)
        row.names(perf) = NULL
        combined = merge(perf, instanceFeatures, by = data$ids)
        
        # merge with solver features
        combined = merge(combined, algorithmFeatures, by = "algorithm") 
        
        # reshape successes
        colnames(successes) = c(data$ids, gsub("_success", "", snames))
        successes = reshape(successes, direction = "long", varying = pnames, 
                            v.names = "success", timevar = "algorithm", 
                            times = pnames, idvar = instanceCommon)
        row.names(successes) = NULL
        combined = merge(combined, successes, by=c(commonSuccess, "algorithm"))
        snames = "success"
    }
    
    if(is.null(algorithmFeatures)) {
        retval = list(data=combined, features=setdiff(names(instanceFeatures), instanceCommon),
                      performance=pnames, success=snames, minimize=minimize,
                      best=best, ids=instanceCommon)
    } else {
        retval = list(data=combined, features=setdiff(names(instanceFeatures), instanceCommon),
                      algorithmFeatures=setdiff(names(algorithmFeatures), "algorithm"),
                      performance=pnames, success=snames, minimize=minimize,
                      best=best, ids=instanceCommon)
    }
    
    
    if(!is.null(extra)) {
        if(!is.null(algorithmFeatures)) {
            stop("extras can't be linked to rest of data -- not yet implemented with algorithmic features!")
        }
        extraNames = setdiff(names(extra), instanceCommon)
        combined = merge(combined, extra, by=instanceCommon)
        retval$extra = extraNames
        retval$data = combined
    }
    
    #FIXME: implement algorithm feature costs
    if(!is.null(costs)) {
        if(is.numeric(costs)) {
            # same cost for everything
            combined$cost = rep.int(costs, times=length(best))
            cnames = c("cost")
        } else if(is.data.frame(costs)) {
            commonCosts = intersect(names(costs), instanceCommon)
            if(length(commonCosts) == 0) {
                stop("Costs can't be linked to rest of data -- no common columns!")
            }
            cnames = setdiff(names(costs), instanceCommon)
            if(length(intersect(cnames, retval$features)) != length(retval$features)) {
                stop(paste("Costs (", paste(cnames, collapse=", "), ") and features (", paste(retval$features, collapse=", "), ") are disjoint!", sep=""))
            }
            cnames = as.vector(sapply(cnames, function(x) { paste(x, "cost", sep="_") }))
            names(costs) = c(commonCosts, cnames)
            combined = merge(combined, costs, by=instanceCommon)
        } else if(is.list(costs)) {
            # cost groups
            dups = intersect(names(costs$groups), retval$features)
            if(length(dups) > 0) {
                stop(paste("Some cost groups have the same names as features:", paste(dups, collapse=", ")))
            }
            retval$costGroups = costs$groups
            commonCosts = intersect(names(costs$values), instanceCommon)
            cnames = setdiff(names(costs$values), instanceCommon)
            if(length(commonCosts) == 0) {
                stop("Costs can't be linked to rest of data -- no common columns!")
            }
            if(length(intersect(do.call(c, costs$groups), retval$features)) != length(retval$features)) {
                stop(paste("Cost groups (", paste(do.call(c, costs$groups), collapse=", "), ") and features (", paste(retval$features, collapse=", "), ") are disjoint!", sep=""))
            }
            if(length(intersect(names(costs$groups), cnames)) != length(names(costs$groups))) {
                stop(paste("Cost groups (", paste(do.call(c, costs$groups), collapse=", "), ") and costs (", paste(cnames, collapse=", "), ") are disjoint!", sep=""))
            }
            combined = merge(combined, costs$values, by=instanceCommon)
        } else {
            stop("Invalid format for costs!")
        }
        retval$cost = cnames
        retval$data = combined
    }
    
    class(retval) = "llama.data"
    return(retval)
}
