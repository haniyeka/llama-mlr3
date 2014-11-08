misclassificationPenalties <-
function(data=NULL, model=NULL, addCosts=NULL) {
    if(is.null(data) || is.null(model)) {
        stop("Need both data and model to calculate misclassification penalties!")
    }

    if(attr(model, "hasPredictions")) {
        predictions = model$predictions
    } else {
        if(length(data$test) > 0) {
            predictions = do.call(rbind, lapply(data$test, function(x) {
                data$data = data$data[x,]
                model(data)
            }))
        } else {
            predictions = model(data)
        }
    }

    optfun = if(data$minimize) { min } else { max }
    perfs = data$data[data$performance]
    opts = apply(perfs, 1, optfun)
    chosen = as.character(unlist(tapply(predictions$algorithm, predictions[data$ids],
        head, 1, simplify = FALSE)))
    sapply(1:nrow(perfs), function(i) {
        if(is.na(chosen[i])) {
            0
        } else {
            as.numeric(abs(perfs[i,chosen[i]] - opts[i]))
        }
    })
}
