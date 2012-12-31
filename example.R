source("R/input.R")
data = cptk.read("data/features_example.csv", "data/times_example.csv", "data/timeouts_example.csv")

raw = read.csv("data/features_example.csv")

source("R/process.R")
source("R/evaluate.R")

source("R/classify.R")
require(RWeka)
set.seed(1)
res = cptk.classify(classifier=J48, data=data, split=cptk.cvFolds, eval=list(cptk.successes, cptk.misclassificationPenalty, cptk.parscore))
print(res$eval)
#print(res$predictor(raw))

source("R/cluster.R")
res = cptk.cluster(clusterer=make_Weka_clusterer("weka/clusterers/EM"), data=data, split=cptk.cvFolds, pre=cptk.normalize, eval=cptk.parscore)
print(res$eval)
#print(res$predictor(raw))

source("R/regression.R")
res = cptk.regression(regressor=make_Weka_classifier("weka/classifiers/functions/LinearRegression"), data=data, split=cptk.cvFolds, eval=cptk.misclassificationPenalty)
print(res$eval)
#print(res$predictor(raw))
