parscores <-
function(data=NULL, model=NULL, factor=10, timeout=NULL) {
    if(is.null(data) || is.null(model)) {
        stop("Need both data and model to calculate PAR scores!")
    }
    if(is.null(data$success)) {
        stop("Need successes to calculate PAR scores!")
    }

    if(is.function(model)) {
        # vbs or a model predictor
        if(length(data$test) > 0) {
            # we've passed in splits
            stop("The model you've passed in only works on the full data.")
        }
        predictions = model(data)
    } else {
        # it's a model
        predictions = model$predictions
    }
    if(length(data$test) == 0) {
        data$test = list(data$data)
        predictions = list(predictions)
    }
    return(unlist(lapply(1:length(data$test), function(i) {
        sapply(1:nrow(data$test[[i]]), function(j) {
            perfs = subset(data$test[[i]][j,], T, data$performance)
            successes = subset(data$test[[i]][j,], T, data$success)
            score = as.numeric(perfs[which(data$performance == predictions[[i]][[j]]$algorithm[1])])
            if(!as.logical(successes[which(data$performance == predictions[[i]][[j]]$algorithm[1])])) {
                score = (if(is.null(timeout)) { score } else { timeout }) * factor
            }
            score
            })
        })))
}
