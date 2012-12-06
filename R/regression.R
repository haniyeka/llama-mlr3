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

#classifiers <- list(
#                 list(name="REPTree", func=function(x, y) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.trees.REPTree")) }),
#                 list(name="LinearRegression", func=function(x, y) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.functions.LinearRegression")) }),
#                 list(name="GaussianProcesses", func=function(x, y) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.functions.GaussianProcesses")) }),
#                 list(name="ZeroR", func=function(x, y) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.rules.ZeroR")) }),
#                 list(name="SMOreg", func=function(x, y) { make_Weka_classifier("weka/classifiers/meta/AttributeSelectedClassifier")(x, data=y, control=Weka_control(E="weka.attributeSelection.CfsSubsetEval", S="weka.attributeSelection.BestFirst -D 1 -N 5", W="weka.classifiers.functions.SMOreg")) }))

classifiers <- list(
                 list(name="SVM epsilon default", func=function(x, y) { (make_Weka_classifier("weka/classifiers/functions/LibSVM"))(x, data=y, control=Weka_control(S=3)) }),
                 list(name="SVM nu default", func=function(x, y) { (make_Weka_classifier("weka/classifiers/functions/LibSVM"))(x, data=y, control=Weka_control(S=4)) }),
                 list(name="REPTree", func=function(x, y) { (make_Weka_classifier("weka/classifiers/trees/REPTree"))(x, data=y) }),
                 list(name="LinearRegression", func=function(x, y) { (make_Weka_classifier("weka/classifiers/functions/LinearRegression"))(x, data=y) }),
                 list(name="M5Rules", func=function(x, y) { (make_Weka_classifier("weka/classifiers/rules/M5Rules"))(x, data=y) }),
                 list(name="M5P", func=function(x, y) { (make_Weka_classifier("weka/classifiers/trees/M5P"))(x, data=y) }),
                 list(name="AdditiveRegression", func=function(x, y) { (make_Weka_classifier("weka/classifiers/meta/AdditiveRegression"))(x, data=y) }),
                 list(name="GaussianProcesses", func=function(x, y) { (make_Weka_classifier("weka/classifiers/functions/GaussianProcesses"))(x, data=y) }),
                 list(name="ZeroR", func=function(x, y) { (make_Weka_classifier("weka/classifiers/rules/ZeroR"))(x, data=y) }),
                 list(name="SMOreg", func=function(x, y) { (make_Weka_classifier("weka/classifiers/functions/SMOreg"))(x, data=y) }))

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
    library(RWeka)
    library(rJava)
    options(java.parameters = "-Xmx1g")
    .jaddClassPath("/usr/share/java/libsvm.jar")
    totalpen = 0
    solved = 0
    minpen = 0
    minsolved = 0
    for (i in c(1:nfolds)) {
        trainData = subset(folds, folds$fold != i, names(data))
        testData = subset(folds, folds$fold == i, names(data))

        features = subset(names(trainData), c(F, rep.int(F, length(algNames)), rep.int(T, length(names(trainData))-length(algNames)-1)))
        times = subset(names(trainData), c(F, rep.int(T, length(algNames)), rep.int(F, length(names(trainData))-length(algNames)-1)))

        predictions = matrix(nrow=nrow(testData), ncol=length(times))

        for (i in c(1:length(times))) {
            model = classifier$func(trainData[[times[i]]]~., subset(trainData, T, features))
            predictions[,i] = predict(model, subset(testData, T, features))
        }

        tmp = apply(predictions, 1, function(x) { head(which(x %in% min(x)), 1) })

        for (id in c(1:nrow(testData))) {
            thistime = subset(testData[id,], T, times)[tmp[id]]
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
        #for (id in c(1:nrow(testData))) {
        #    totalpen = totalpen + as.numeric(subset(testData[id,], T, times)[tmp[id]] - min(subset(testData[id,], T, times)))
        #}
        #cat(paste(fold, totalpen, sep=", "), sep="\n")
    }
    #cat(paste(classifier$name, totalpen, sep=", "), sep="\n")
    cat(paste(classifier$name, minpen, minsolved, totalpen, solved, sep=", "), sep="\n")
}
