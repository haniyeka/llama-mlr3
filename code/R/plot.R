perfScatterPlot <-
function(metric, modelx, modely, datax, datay=datax, addCostsx=NULL, addCostsy=NULL, pargs=NULL, ...) {
    if(is.null(metric)) {
        stop("Need evaluation metric for plotting!")
    }
    if(is.null(modelx) || is.null(modely)) {
        stop("Need models to plot performances!")
    }
    if(is.null(datax)) {
        stop("Need data to plot performances!")
    }

    idCols = intersect(datax$ids, datay$ids)
    if(length(idCols) == 0) {
        stop("Cannot match up the two data frames!")
    }

    if(length(datax$test) > 0) {
        idsx = do.call(rbind, lapply(datax$test, function(x) {
            datax$data[x,idCols,drop=F]
        }))
    } else {
        idsx = datax$data[idCols]
    }
    if(length(datay$test) > 0) {
        idsy = do.call(rbind, lapply(datay$test, function(x) {
            datay$data[x,idCols,drop=F]
        }))
    } else {
        idsy = datay$data[idCols]
    }

    scoresx = cbind(idsx, data.frame(scorex=metric(datax, modelx, addCosts=addCostsx, ...)))
    scoresy = cbind(idsy, data.frame(scorey=metric(datay, modely, addCosts=addCostsy, ...)))

    d = merge(scoresx, scoresy, by=idCols)
    ggplot(d, aes_string(x="scorex", y="scorey")) +
        geom_abline(slope = 1) +
        geom_point(pargs)
}
