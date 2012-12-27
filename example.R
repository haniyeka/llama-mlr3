source("R/input.R")
data = cptk.read("data/features_example.csv", "data/times_example.csv")

source("R/process.R")
splitdata = cptk.trainTest(data)

source("R/classify.R")
source("R/evaluate.R")
require(RWeka)
misclass = cptk.classify(J48, splitdata$train, splitdata$test, data$features, data$performance, cptk.misclassificationPenalty)
print(misclass)
