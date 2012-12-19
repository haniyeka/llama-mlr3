require('doMC')

registerDoMC()

nfolds = 10

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

data = read.table(tail(commandArgs(), 1), header=F, sep=",", na.strings="?")
data$V1 = as.factor(data$V1)

set.seed(1)

require(plyr)
createFolds = function(x, k) {
    n = nrow(x)
    x$fold = rep(1:k,length.out = n)[sample(n,n)]
    x
}
folds = ddply(data, .(V1), createFolds, k = nfolds)

normalize = function(d, minses, maxes) {
    sweep(d, 2, minses, "-")
    sweep(d, 2, maxes-minses, "/")
    sweep(d, 2, 2, "*")
    sweep(d, 2, 1, "-")
}

res = foreach (i = c(1:nfolds), .combine = function(x, y) { c(x[1] + y[1], x[2] + y[2]) }) %dopar% {
    options(java.parameters = "-Xmx1g")
    require('RWeka')
    require('rJava')

    penalty = 0
    solved = 0
    trainData = subset(folds, folds$fold != i, names(data))
    testData = subset(folds, folds$fold == i, names(data))

    features = subset(names(trainData), c(F, rep.int(F, length(algNames)), rep.int(T, length(names(trainData))-length(algNames)-1)))
    times = subset(names(trainData), c(F, rep.int(T, length(algNames)), rep.int(F, length(names(trainData))-length(algNames)-1)))

    minses = apply(subset(trainData, T, features), 2, min, na.rm=T)
    maxes = apply(subset(trainData, T, features), 2, max, na.rm=T)

    model = make_Weka_clusterer("weka/clusterers/EM")(normalize(subset(trainData, T, features), minses, maxes))
    trainclusters = predict(model)
    bestalgs = by(trainData, trainclusters, function(x) { names(sort(table(x$V1), decreasing=T))[1] })

    testclusters = predict(model, normalize(subset(testData, T, features), minses, maxes))

    for (id in c(1:nrow(testData))) {
        thistime = subset(testData[id,], T, times)[which(algNames == bestalgs[testclusters[id]+1])]
        if(length(thistime) == 0 || thistime >= 3600) {
            thistime = 36000
        } else {
            solved = solved + 1
        }
        penalty = penalty + as.numeric(thistime)
    }
    return(c(penalty, solved))
}
cat(cat(res[1]/nrow(data), res[2], sep=", "), sep="\n")
