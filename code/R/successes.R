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
    chosen = as.vector(by(predictions$algorithm, predictions[c(data$ids, "iteration")], function(ss) {
        idx = which(as.character(ss[1]) == data$performance)
        if(length(idx) == 0) { # NA prediction
            NA
        } else {
            idx
        }
    }))
    if(!addCosts || is.null(data$cost)) {
        costs = rep.int(0, nrow(perfs))
    } else {
        costs = apply(data$data[usedFeatures], 1, sum)
    }
    sapply(1:length(chosen), function(i) {
        if(is.na(chosen[i])) {
            FALSE
        } else {
            ids = predictions[i,][data$ids]
            idx = which(apply(data$data[data$ids], 1, function(x) { x == ids }))
            value = as.numeric(perfs[idx,chosen[i]]) + costs[idx]
            if(!as.logical(successes[idx,chosen[i]]) || value > timeout) {
                FALSE
            } else {
                TRUE
            }
        }
    })
}
