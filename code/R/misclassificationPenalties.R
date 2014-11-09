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
                data$best = data$best[x]
                model(data)
            }))
        } else {
            predictions = model(data)
        }
    }

    optfun = if(data$minimize) { min } else { max }
    perfs = data$data[data$performance]
    opts = apply(perfs, 1, optfun)
    chosen = as.vector(by(predictions$algorithm, predictions[c(data$ids, "iteration")], function(ss) {
        idx = which(as.character(ss[1]) == data$performance)
        if(length(idx) == 0) { # NA prediction
            NA
        } else {
            idx
        }
    }))
    ids = predictions[data$ids]
    idxs = which(apply(ids, 1, function(id) { any(data$data[data$ids] == id) }))
    sapply(1:length(chosen), function(i) {
        if(is.na(chosen[i])) {
            0
        } else {
            as.numeric(abs(perfs[idxs[i],chosen[i]] - opts[idxs[i]]))
        }
    })
}
