perfScatterPlot <-
function(metric, modelx, modely, datax, datay=datax, addCostsx=NULL, addCostsy=NULL, ...) {
    if(is.null(metric)) {
        stop("Need evaluation metric for plotting!")
    }
    if(is.null(modelx) || is.null(modely)) {
        stop("Need models to plot performances!")
    }
    if(is.null(datax)) {
        stop("Need data to plot performances!")
    }

    if(length(datax$test) > 0) {
        idsx = do.call(rbind, lapply(datax$test, function(x) {
            datax$data[x,data$ids,drop=F]
        }))
    } else {
        idsx = datax$data[data$ids]
    }
    if(length(datay$test) > 0) {
        idsy = do.call(rbind, lapply(datay$test, function(x) {
            datay$data[x,data$ids,drop=F]
        }))
    } else {
        idsy = datay$data[data$ids]
    }

    scoresx = cbind(idsx, data.frame(scorex=metric(datax, modelx, addCosts=addCostsx, ...)))
    scoresy = cbind(idsy, data.frame(scorey=metric(datay, modely, addCosts=addCostsy, ...)))

    d = merge(scoresx, scoresy, by=data$ids)
    ggplot(d, aes_string(x="scorex", y="scorey")) + geom_abline(slope = 1)
}
