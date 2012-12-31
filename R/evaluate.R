cptk.misclassificationPenalty = function(allperf, chosenperf, chosensuccess, minimize=T) {
    optfun = if(minimize) { min } else { max }
    return(abs(chosenperf - optfun(allperf)))
}

cptk.successes = function(allperf, chosenperf, chosensuccess) {
    return(as.integer(chosensuccess))
}

cptk.parscore = function(allperf, chosenperf, chosensuccess, factor=10) {
    score = chosenperf
    if(!chosensuccess) { score = score * factor }
    return(score)
}
