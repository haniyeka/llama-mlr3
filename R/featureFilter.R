featureFilter <-
function(filter=NULL, data=NULL) {
    data$original_features = data$features
    b = data$data$best
    data$features = filter(b~., subset(data$data, T, data$features))
    return(data)
}
