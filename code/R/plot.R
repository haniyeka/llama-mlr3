perfScatterPlot <-
function(metric, modelx, modely, datax, datay=datax, addCostsx=TRUE, addCostsy=TRUE, ...) {
    if(is.null(metric)) {
        stop("Need evaluation metric for plotting!")
    }
    if(is.null(modelx) || is.null(modely)) {
        stop("Need models to plot performances!")
    }
    if(is.null(datax)) {
        stop("Need data to plot performances!")
    }

    idCols = setdiff(names(datax$data),
        Reduce('union', list(datax$features, datax$performance, datax$success, datax$cost, "best")))
    if(!all(idCols %in% names(datay$data))) {
        stop("Cannot match up the two data frames!")
    }

    if(length(datax$test) > 0) {
        idsx = Reduce('rbind', datax$test)[idCols]
    } else {
        idsx = datax$data[idCols]
    }
    if(length(datay$test) > 0) {
        idsy = Reduce('rbind', datay$test)[idCols]
    } else {
        idsy = datay$data[idCols]
    }

    scoresx = cbind(idsx, data.frame(scorex=metric(datax, modelx, addCosts=addCostsx, ...)))
    scoresy = cbind(idsy, data.frame(scorey=metric(datay, modely, addCosts=addCostsy, ...)))

    d = merge(scoresx, scoresy, by=idCols)
    ggplot(d, aes_string(x="scorex", y="scorey")) + geom_abline(slope = 1)
}
