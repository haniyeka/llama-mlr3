cptk.parscore <-
function(allperf, chosenperf, chosensuccess, factor=10) {
    score = chosenperf
    if(!chosensuccess) { score = score * factor }
    return(score)
}
