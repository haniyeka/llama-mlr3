misclassificationPenalties <-
function(data=NULL, model=NULL) {
    if(is.null(data) || is.null(model)) {
        stop("Need both data and model to calculate misclassification penalties!")
    }

    if(is.function(model)) {
        # vbs or a model predictor
        if(length(data$test) > 0) {
            predictions = lapply(data$test, function(x) {
                data$data = x
                model(data)
            })
        } else {
            predictions = model(data)
        }
    } else {
        # it's a model
        predictions = model$predictions
    }

    if(length(data$test) == 0) {
        data$test = list(data$data)
        predictions = list(predictions)
    }

    optfun = if(data$minimize) { min } else { max }

    return(unlist(lapply(1:length(data$test), function(i) {
        perfs = data$test[[i]][data$performance]
        opts = apply(perfs, 1, optfun)
        chosen = sapply(predictions[[i]], function(x) { as.character(x$algorithm[1]) })
        sapply(1:nrow(data$test[[i]]), function(j) {
            if(is.na(chosen[j])) {
                NA
            } else {
                as.numeric(abs(perfs[j,chosen[j]] - opts[j]))
            }
        })
    })))
}
