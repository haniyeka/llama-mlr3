successes <-
function(data=NULL, model=NULL, timeout=NULL, addCosts=NULL) {
    if(is.null(data) || is.null(model)) {
        stop("Need both data and model to calculate successes!")
    }
    if(is.null(data$success)) {
        stop("Need successes to calculate successes!")
    }

    if(is.null(addCosts)) {
        ac = attr(model, "addCosts")
        if(is.null(ac) || ac == TRUE) {
            addCosts = TRUE
        } else {
            addCosts = FALSE
        }
    }

    hp = attr(model, "hasPredictions")
    if(is.null(hp) || hp != TRUE) {
        if(length(data$test) > 0) {
            predictions = do.call(rbind, lapply(data$test, function(x) {
                data$data = data$data[x,]
                data$best = data$best[x]
                model(data)
            }))
        } else {
            predictions = model(data)
        }
    } else {
        predictions = model$predictions
    }

    if(is.null(timeout)) {
        # if timeout value wasn't given, assume maximum from data set
        timeout = max(data$data[data$performance])
    }

    if(addCosts || !is.null(data$costs)) {
        if(is.null(data$costGroups)) {
            usedFeatures = intersect(data$cost, sapply(data$features, function(x) { paste(x, "cost", sep="_") }))
        } else {
            usedFeatures = subset(data$cost, sapply(data$cost, function(x) { length(intersect(data$costGroups[[x]], data$features)) > 0 }))
        }
    }

    perfs = data$data[data$performance]
    successes = data$data[data$success]
    if(!addCosts || is.null(data$cost)) {
        costs = rep.int(0, nrow(perfs))
    } else {
        costs = apply(data$data[usedFeatures], 1, sum)
    }
    as.vector(na.omit(as.vector(by(predictions, predictions[c(data$ids, "iteration")], function(ss) {
        pid = which(as.character(ss$algorithm[1]) == data$performance)
        if(length(pid) == 0) { # NA prediction
            FALSE
        } else {
            iid = which(ss[1,data$ids] == data$data[data$ids])
            value = as.numeric(perfs[iid,pid]) + costs[iid]
            if(!as.logical(successes[iid,pid]) || value > timeout) {
                FALSE
            } else {
                TRUE
            }
        }
    }))))
}
