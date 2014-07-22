require(llama)

data(satsolvers)
data = satsolvers

folds = cvFolds(satsolvers)

model = classify(makeLearner("classif.J48"), folds)
mean(parscores(folds, model))
sum(successes(folds, model))

model$predictions[[1]][[1]]

#####

mean(parscores(data, vbs))
sum(successes(data, vbs))
mean(parscores(data, singleBest))
sum(successes(data, singleBest))

#####

model = regression(makeLearner("regr.lm"), folds)
mean(parscores(folds, model))
sum(successes(folds, model))

model$predictions[[1]][[1]]

#####

model = classifyPairs(makeLearner("classif.J48"), folds)
mean(parscores(folds, model))
sum(successes(folds, model))

model$predictions[[1]][[1]]

#####

model = cluster(makeLearner("cluster.XMeans"), folds)
mean(parscores(folds, model))
sum(successes(folds, model))

model$predictions[[1]][[1]]

#####

require(FSelector)
newdata = featureFilter(cfs, data)

newfolds = cvFolds(newdata)
model = classify(makeLearner("classif.J48"), folds)
newmodel = classify(makeLearner("classif.J48"), newfolds)
mean(parscores(folds, model))
sum(successes(folds, model))
mean(parscores(newfolds, newmodel))
sum(successes(newfolds, newmodel))

#####

model = cluster(makeLearner("cluster.XMeans"), folds, pre=normalize)
mean(parscores(folds, model))
sum(successes(folds, model))

#####

imputed = imputeCensored(data, makeLearner("regr.lm"))
imputedFolds = cvFolds(imputed)
oldmodel = regression(makeLearner("regr.lm"), folds)
model = regression(makeLearner("regr.lm"), imputedFolds)
mean(misclassificationPenalties(folds, oldmodel))
mean(misclassificationPenalties(imputedFolds, model))

#####

sort(contributions(satsolvers))

#####

ensembleModel = classify(list(makeLearner("classif.J48"), makeLearner("classif.ksvm"), makeLearner("classif.knn")), folds)
stackedEnsembleModel = classify(list(makeLearner("classif.J48"), makeLearner("classif.ksvm"), makeLearner("classif.knn"), .combine=makeLearner("classif.J48")), folds)
mean(parscores(folds, ensembleModel))
sum(successes(folds, ensembleModel))
mean(parscores(folds, stackedEnsembleModel))
sum(successes(folds, stackedEnsembleModel))

ensembleModel$predictions[[1]][[1]]

#####

model = regression(makeLearner("regr.lm"), folds)
stackedModel = regression(makeLearner("regr.lm"), folds, combine=makeLearner("classif.OneR"))
mean(parscores(folds, model))
sum(successes(folds, model))
mean(parscores(folds, stackedModel))
sum(successes(folds, stackedModel))

#####

stackedModel = regression(makeLearner("regr.lm"), folds, combine=makeLearner("classif.OneR"),
    expand=function(x) {
        cbind(x, combn(c(1:ncol(x)), 2, function(y) {
            abs(x[,y[1]] - x[,y[2]])
        }))
    })
mean(parscores(folds, stackedModel))
sum(successes(folds, stackedModel))

#####

pdf("times.pdf")
par(mar=c(7,1,3,1))
times = subset(satsolvers$data, T, satsolvers$performance)
cols = gray(seq(1, 0, length.out=255))
image(t(as.matrix(times)), axes=F, col=cols)
axis(1, labels=satsolvers$performance, at=seq(0, 1, 1/(length(satsolvers$performance)-1)), las=2)
legend("top", legend=c(min(times), max(times)), fill=c("white", "black"), bty="n", inset=-0.12, xpd=NA)
dev.off()

pdf("times-log.pdf")
par(mar=c(7,1,3,1))
image(log10(t(as.matrix(times))), axes=F, col=cols)
axis(1, labels=satsolvers$performance, at=seq(0, 1, 1/(length(satsolvers$performance)-1)), las=2)
legend("top", legend=c(min(times), max(times)), fill=c("white", "black"), bty="n", inset=-0.12, xpd=NA)
dev.off()

pdf("rank.pdf")
par(mar=c(7,1,3,1))
image(apply(times, 1, order), axes=F, col=cols)
axis(1, labels=satsolvers$performance, at=seq(0, 1, 1/(length(satsolvers$performance)-1)), las=2)
legend("top", legend=c(1, length(satsolvers$performance)), fill=c("white", "black"), bty="n", inset=-0.12, xpd=NA)
dev.off()

pdf("features.pdf")
par(mar=c(7,1,3,1))
features = subset(satsolvers$data, T, satsolvers$features)
par(mar=c(10,1,3,1))
image(t(as.matrix(features)), axes=F, col=cols)
axis(1, labels=satsolvers$features, at=seq(0, 1, 1/(length(satsolvers$features)-1)), las=2)
legend("top", legend=c(min(features), max(features)), fill=c("white", "black"), bty="n", inset=-0.12, xpd=NA)
dev.off()

pdf("features-norm.pdf")
par(mar=c(7,1,3,1))
nFeatures = normalize(features)
par(mar=c(10,1,1,1))
image(t(as.matrix(nFeatures$features)), axes=F, col=cols)
axis(1, labels=satsolvers$features, at=seq(0, 1, 1/(length(satsolvers$features)-1)), las=2)
dev.off()

pdf("classification.pdf")
par(mar=c(7,1,3,1))
model = classify(makeLearner("classif.J48"), folds)
scores = parscores(folds, model)
par(mar=c(1,1,1,10))
image(log10(t(as.matrix(scores))), axes=F, col=cols)
legend("right", legend=quantile(scores), fill=gray(seq(1, 0, length.out=5)), bty="n", inset=-0.3, xpd=NA)
dev.off()

pdf("comparison.pdf")
par(mar=c(7,1,3,1))
model1 = classify(makeLearner("classif.J48"), folds)
model2 = regression(makeLearner("regr.lm"), folds)
model3 = classifyPairs(makeLearner("classif.J48"), folds)
model4 = regressionPairs(makeLearner("regr.lm"), folds)
model5 = cluster(makeLearner("cluster.XMeans"), folds)
scores1 = parscores(folds, model1)
scores2 = parscores(folds, model2)
scores3 = parscores(folds, model3)
scores4 = parscores(folds, model4)
scores5 = parscores(folds, model5)
par(mar=c(10,1,1,1))
names = c(paste("classify", "J48", mean(scores1), sep="\n"), paste("regression", "lm", mean(scores2), sep="\n"), paste("classifyPairs", "J48", mean(scores3), sep="\n"), paste("regressionPairs", "lm", mean(scores4), sep="\n"), paste("cluster", "XMeans", mean(scores5), sep="\n"))
scores = c(scores1, scores2, scores3, scores4, scores5)
image(log10(t(matrix(scores, ncol=5))), axes=F, col=cols)
axis(1, labels=names, at=seq(0, 1, 1/(length(names)-1)), las=1, tick=F, line=1)
legend("bottom", legend=quantile(scores), fill=gray(seq(1, 0, length.out=5)), bty="n", inset=-0.4, xpd=NA)
dev.off()

pdf("rank-diff.pdf")
par(mar=c(7,1,1,5))
ranks = apply(sapply(unlist(model2$predictions, recursive=F), function(x) { x$algorithm }), 2, function(y) { as.vector(sapply(y, function(z) { which(z == satsolvers$performance) })) })
diffs = apply(times, 1, order)-ranks
cols = c(colorRampPalette(c("red", "white"))(128), colorRampPalette(c("white", "blue"))(128))
breaks = c(seq(min(diffs), -1/128, length.out=128), 0, seq(1/128, max(diffs), length.out=128))
image(diffs, axes=F, col=cols, breaks=breaks)
axis(1, labels=satsolvers$performance, at=seq(0, 1, 1/(length(satsolvers$performance)-1)), las=2)
legend("right", legend=quantile(diffs), fill=cols[quantile(1:length(cols))], bty="n", inset=-0.15, xpd=NA)
dev.off()
