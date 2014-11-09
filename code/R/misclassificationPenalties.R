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
    as.vector(na.omit(as.vector(by(predictions, predictions[c(data$ids, "iteration")], function(ss) {
        pid = which(as.character(ss$algorithm[1]) == data$performance)
        if(length(pid) == 0) { # NA prediction
            0
        } else {
            iid = which(ss[1,data$ids] == data$data[data$ids])
            as.numeric(abs(as.numeric(perfs[iid,pid]) - opts[iid]))
        }
    }))))
}
