suppressWarnings(suppressMessages({
    library(doMC)
}))

registerDoMC()

nfolds = 10

path = tail(commandArgs(), 1)

algNames = c("direct-minisat",
"direct-cryptominisat",
"direct-glucose",
"direct-minisat_noelim",
"direct-lingeling",
"direct-clasp",
"direct-picosat",
"direct-precosat",
"direct-kcnfs",
"direct-march_rw",
"direct-mxc",
"direct-riss",
"direct-sat4j",
"direct-glueminisat",
"direct-qutersat",
"direct-contrasat",
"direct-MPhaseSAT64",
"direct-cirminisat",
"direct-rsat",
"order-minisat",
"order-cryptominisat",
"order-glucose",
"order-minisat_noelim",
"order-lingeling",
"order-clasp",
"order-picosat",
"order-precosat",
"order-kcnfs",
"order-march_rw",
"order-mxc",
"order-riss",
"order-sat4j",
"order-glueminisat",
"order-qutersat",
"order-contrasat",
"order-MPhaseSAT64",
"order-cirminisat",
"order-rsat",
"support-minisat",
"support-cryptominisat",
"support-glucose",
"support-minisat_noelim",
"support-lingeling",
"support-clasp",
"support-picosat",
"support-precosat",
"support-kcnfs",
"support-march_rw",
"support-mxc",
"support-riss",
"support-sat4j",
"support-glueminisat",
"support-qutersat",
"support-contrasat",
"support-MPhaseSAT64",
"support-cirminisat",
"support-rsat")

#classifiers = list(list(name="J48graft", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.trees.J48graft")) }),
#                 list(name="SimpleLogistic", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.functions.SimpleLogistic")) }),
#                 list(name="BayesNet", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.bayes.BayesNet")) }),
#                 list(name="ConjunctiveRule", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.rules.ConjunctiveRule")) }),
#                 list(name="LADTree", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.trees.LADTree")) }),
#                 list(name="HyperPipes", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.misc.HyperPipes")) }),
#                 list(name="FT", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.trees.FT")) }),
#                 list(name="PART", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.rules.PART")) }),
#                 list(name="JRip", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.rules.JRip")) }),
#                 list(name="RandomTree", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.trees.RandomTree")) }),
#                 list(name="RandomForest", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.trees.RandomForest")) }),
#                 list(name="BFTree", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.trees.BFTree")) }),
#                 list(name="REPTree", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.trees.REPTree")) }),
#                 list(name="DecisionTable", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.rules.DecisionTable")) }),
#                 list(name="IBk", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.lazy.IBk")) }),
#                 list(name="MultilayerPerceptron", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.functions.MultilayerPerceptron")) }),
#                 list(name="OneR", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.rules.OneR")) }),
#                 list(name="ZeroR", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, weights=z, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.rules.ZeroR")) }))

classifiers = list(
                 list(name="J48", func=function(x, y, z) { J48(x, data=y, weights=z, control=Weka_control(N=3, R=TRUE)) }),
                 list(name="J48graft", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/trees/J48graft")(x, data=y, weights=z, control=Weka_control(N=3, R=TRUE)) }),
                 list(name="BayesNet", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/bayes/BayesNet")(x, data=y, weights=z) }),
                 list(name="SimpleLogistic", func=function(x, y, z) { make_Weka_classifier("weka/classifiers/functions/SimpleLogistic")(x, data=y, weights=z) }),
                 list(name="PART", func=function(x, y, z) { PART(x, data=y, weights=z, control=Weka_control(U=TRUE)) }),
                 list(name="JRip", func=function(x, y, z) { JRip(x, data=y, weights=z, control=Weka_control(P=TRUE)) }),
                 list(name="RandomTree", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/trees/RandomTree"))(x, data=y, weights=z) }),
                 list(name="RandomForest", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/trees/RandomForest"))(x, data=y, weights=z) }),
                 list(name="REPTree", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/trees/REPTree"))(x, data=y, weights=z) }),
                 list(name="DecisionTable", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/rules/DecisionTable"))(x, data=y, weights=z, control=Weka_control(E="acc")) }),
                 list(name="IBk", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/lazy/IBk"))(x, data=y, weights=z, control=Weka_control(I=TRUE)) }),
                 list(name="IBk3", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/lazy/IBk"))(x, data=y, weights=z, control=Weka_control(K=3, I=TRUE)) }),
                 list(name="IBk5", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/lazy/IBk"))(x, data=y, weights=z, control=Weka_control(K=5, I=TRUE)) }),
                 list(name="IBk10", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/lazy/IBk"))(x, data=y, weights=z, control=Weka_control(K=10, I=TRUE)) }),
                 list(name="MultilayerPerceptron", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/functions/MultilayerPerceptron"))(x, data=y, weights=z) }),
                 list(name="OneR", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/rules/OneR"))(x, data=y, weights=z) }),
                 list(name="AdaBoost", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/meta/AdaBoostM1"))(x, data=y, weights=z, control=Weka_control(I=100, Q=TRUE)) }),
                 list(name="SVM rbf", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/functions/LibSVM"))(x, data=y, weights=z, control=Weka_control(K=2)) }),
                 list(name="SVM sigmoid", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/functions/LibSVM"))(x, data=y, weights=z, control=Weka_control(K=3)) }),
                 list(name="ZeroR", func=function(x, y, z) { (make_Weka_classifier("weka/classifiers/rules/ZeroR"))(x, data=y, weights=z) }))

data = read.table(path, header=F, sep=",")
data$V1 = as.factor(data$V1)

set.seed(1)

require(plyr)
createFolds = function(x,k){
    n = nrow(x)
    x$fold = rep(1:k,length.out = n)[sample(n,n)]
    x
}

folds = ddply(data, .(V1), createFolds, k = nfolds)

foreach (classifier = classifiers) %dopar% {
    options(java.parameters = "-Xmx1g")
    library(RWeka)
    library(rJava)
    .jaddClassPath("/usr/share/java/libsvm.jar")
    WPM("load-package", "LibSVM")
    WPM("load-package", "J48graft")
    totalpen = 0
    trainpen = 0
    solved = 0
    minpen = 0
    minsolved = 0
    foreach (i = c(1:nfolds)) %do% {
        trainData = subset(folds, folds$fold != i, names(data))
        testData = subset(folds, folds$fold == i, names(data))

        features = subset(names(trainData), c(F, rep.int(F, length(algNames)), rep.int(T, length(names(trainData))-length(algNames)-1)))
        times = subset(names(trainData), c(F, rep.int(T, length(algNames)), rep.int(F, length(names(trainData))-length(algNames)-1)))

        #model = classifier$func(trainData$V1~., subset(trainData, T, features), rep.int(1, length(trainData$V1)))
        model = classifier$func(trainData$V1~., subset(trainData, T, features), apply(subset(trainData, T, times), 1, function(x) { if(max(x) == 3600) { 36000 } else { max(x) } }) - apply(subset(trainData, T, times), 1, min))
        predictions = predict(model, subset(testData, T, features))

        for (id in c(1:nrow(testData))) {
            thistime = subset(testData[id,], T, times)[which(algNames==predictions[id])]
            if(thistime == 3600) {
                thistime = 36000
            } else {
                solved = solved + 1
            }
            totalpen = totalpen + thistime
            thismin = min(subset(testData[id,], T, times))
            if(thismin == 3600) {
                thismin = 36000
            } else {
                minsolved = minsolved + 1
            }
            minpen = minpen + thismin
        }
    }
    cat(paste(classifier$name, minpen, minsolved, totalpen, solved, sep=", "), sep="\n")
}
