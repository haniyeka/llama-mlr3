source("R/input.R")
data = cptk.read("data/features_example.csv", "data/times_example.csv", "data/timeouts_example.csv")

source("R/process.R")
source("R/evaluate.R")

source("R/classify.R")
require(RWeka)
set.seed(1)
solved = cptk.classify(classifier=J48, data=data, split=cptk.cvFolds, eval=list(cptk.successes, cptk.misclassificationPenalty))
print(solved)

source("R/cluster.R")
par = cptk.cluster(clusterer=make_Weka_clusterer("weka/clusterers/EM"), data=data, split=cptk.cvFolds, eval=cptk.parscore)
print(par)

source("R/regression.R")
misclass = cptk.regression(regressor=make_Weka_classifier("weka/classifiers/functions/LinearRegression"), data=data, split=cptk.cvFolds, eval=cptk.misclassificationPenalty)
print(misclass)
