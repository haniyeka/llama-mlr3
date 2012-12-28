source("R/input.R")
data = cptk.read("data/features_example.csv", "data/times_example.csv")

source("R/classify.R")
source("R/evaluate.R")
source("R/process.R")
require(RWeka)
set.seed(1)
misclass = cptk.classify(classifier=J48, data=data, split=cptk.cvFolds, eval=cptk.misclassificationPenalty)
print(misclass)

source("R/cluster.R")
misclass = cptk.cluster(clusterer=make_Weka_clusterer("weka/clusterers/EM"), data=data, split=cptk.cvFolds, eval=cptk.misclassificationPenalty)
print(misclass)

source("R/regression.R")
misclass = cptk.regression(regressor=make_Weka_classifier("weka/classifiers/functions/LinearRegression"), data=data, split=cptk.cvFolds, eval=cptk.misclassificationPenalty)
print(misclass)
