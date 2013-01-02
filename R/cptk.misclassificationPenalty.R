cptk.misclassificationPenalty <-
function(allperf, chosenperf, chosensuccess, minimize=T) {
    optfun = if(minimize) { min } else { max }
    return(abs(chosenperf - optfun(allperf)))
}
