successes <-
function(data=NULL, model=NULL, timeout=NULL, addCosts=TRUE) {
    if(is.null(data) || is.null(model)) {
        stop("Need both data and model to calculate successes!")
    }
    if(is.null(data$success)) {
        stop("Need successes to calculate successes!")
    }

    hp = attr(model, "hasPredictions")
    if(is.null(hp) || hp != TRUE) {
        if(length(data$test) > 0) {
            predictions = lapply(data$test, function(x) {
                data$data = x
                model(data)
            })
        } else {
            predictions = model(data)
        }
    } else {
        predictions = model$predictions
    }
    if(length(data$test) == 0) {
        data$test = list(data$data)
        predictions = list(predictions)
    }

    if(is.null(timeout)) {
        # if timeout value wasn't given, assume maximum from data set
        timeout = max(data$data[data$performance])
    }

    if(addCosts || !is.null(data$costs)) {
        if(is.null(data$costGroups)) {
            usedFeatures = intersect(data$cost, sapply(data$features, function(x) { paste(x, "cost", sep="_") }))
        } else {
            usedGroups = subset(data$cost, sapply(data$cost, function(x) { length(intersect(data$costGroups[[x]], data$features)) > 0 }))
        }
    }

    return(unlist(lapply(1:length(data$test), function(i) {
        perfs = data$test[[i]][data$performance]
        successes = data$test[[i]][data$success]
        chosen = sapply(predictions[[i]], function(x) { which(as.character(x$algorithm[1]) == data$performance) })
        if(!addCosts || is.null(data$cost)) {
            costs = rep.int(0, nrow(data$test[[i]]))
        } else {
            if(is.null(data$costGroups)) {
                costs = apply(data$test[[i]][usedFeatures], 1, sum)
            } else {
                costs = apply(data$test[[i]][usedGroups], 1, sum)
            }
        }
        sapply(1:nrow(data$test[[i]]), function(j) {
            if(is.numeric(chosen[j])) {
                value = as.numeric(perfs[j,chosen[j]]) + costs[j]
                if(!as.logical(successes[j,chosen[j]]) || value > timeout) {
                    FALSE
                } else {
                    TRUE
                }
            } else {
                NA
            }
        })
    })))
}
