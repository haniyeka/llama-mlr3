parscores <-
function(data=NULL, model=NULL, factor=10, timeout=NULL, addCosts=NULL) {
    if(is.null(data) || is.null(model)) {
        stop("Need both data and model to calculate PAR scores!")
    }
    if(is.null(data$success)) {
        stop("Need successes to calculate PAR scores!")
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
    chosen = tapply(predictions$algorithm, predictions[data$ids], function(ss) {
        idx = which(as.character(ss[1]) == data$performance)
        if(length(idx) == 0) { # NA prediction
            NA
        } else {
            idx
        }
    })
    if(!addCosts || is.null(data$cost)) {
        costs = rep.int(0, nrow(perfs))
    } else {
        costs = apply(data$data[usedFeatures], 1, sum)
    }
    sapply(1:nrow(perfs), function(i) {
        if(is.na(chosen[i])) {
            timeout * factor
        } else {
            score = as.numeric(perfs[i,chosen[i]]) + costs[i]
            if(!as.logical(successes[i,chosen[i]]) || score > timeout) {
                score = timeout * factor
            }
            score
        }
    })
}
