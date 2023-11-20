contributions <-
function(data=NULL) {
    if(!testClass(data, "llama.data")) {
        stop("Need data to determine contributions!")
    }
  
    #computes the coalition values, then it reduces the values of non-empty coalitions by the maximum value of a single-player coalition.
    #After computing the coalition values, it computes the actual Shapley values
    times = .jarray(as.matrix(subset(data$data, T, data$performance)), dispatch=T)
    coalitionValues = J("shapleyComputation/CoalitionValueCalculator")$computeCoalitionValues(times, data$minimize)
    J("shapleyComputation/CoalitionValueCalculator")$deductFromNonEmptyCoalitionsTheMaxSingletonValue(coalitionValues)
    
    contributions = J("shapleyComputation/ShapleyComputation")$computeShapleyValues(coalitionValues, data$minimize)
    names(contributions) = data$performance

    return(contributions)
}
