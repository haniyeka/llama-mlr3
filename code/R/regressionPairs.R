regressionPairs <-
function(regressor=NULL, data=NULL, pre=function(x, y=NULL) { list(features=x) }, combine=NULL, save.models=NA, use.weights = TRUE) {
    if(!testClass(regressor, "Learner")) {
        stop("Need regressor!")
    }
    assertClass(data, "llama.data")
    hs = attr(data, "hasSplits")
    if(is.null(hs) || hs != TRUE) {
        stop("Need data with train/test split!")
    }
    if(!is.null(combine) && !is.null(data$algorithmFeatures)) {
        stop("Stacking with algorithm features is not supported yet!")
    }
    
    signPair = function(k, p, sign, row, combns) {
        if(p == combns[1,k]) {
            sign(row[k])
        } else if(p == combns[2,k]) {
            sign(-row[k])
        } else {
            0
        }
    }
    
    worstScore = if(data$minimize) { Inf } else { -Inf }
    
    if(is.null(data$algorithmFeatures)) {
        totalBests = data.frame(target=factor(breakBestTies(data), levels=data$performance))
        combns = combn(data$performance, 2)
    } else {
        totalBests = data.frame(target=factor(breakBestTies(data, pairs=TRUE), levels=data$algorithmNames))
        combns = combn(data$algorithmNames, 2)
    }
    
    predictions = rbind.fill(parallelMap(function(i) {
        if(is.null(data$algorithmFeatures)) {
            trf = pre(data$data[data$train[[i]],][data$features])
            tsf = pre(data$data[data$test[[i]],][data$features], trf$meta)
            ids = data$data[data$test[[i]],][data$ids]
            trp = data$data[data$train[[i]],][data$performance]
        } else {
            trf = pre(data$data[data$train[[i]],][c(data$features, data$algos)])
            tsf = pre(data$data[data$test[[i]],][c(data$features, data$algos)], trf$meta)
            ids = unique(data$data[data$test[[i]],][data$ids])
        }
        
        if (is.null(data$algorithmFeatures)) {
            trainpredictions = matrix(nrow=nrow(trf$features), ncol=ncol(combns))
            pairpredictions = matrix(nrow=nrow(tsf$features), ncol=ncol(combns))
            for (j in 1:ncol(combns)) {
                values = data.frame(target=data$data[data$train[[i]],combns[1,j]] - data$data[data$train[[i]],combns[2,j]])
                task = TaskRegr$new(id="regression", target="target", backend=cbind(values, trf$features))
                #, fixup.data="quiet")
                model = regressor$train(task = task)
                if(!is.na(save.models)) {
                    saveRDS(list(model=model, train.data=task, test.data=tsf$features), file = paste(save.models, regressor$id, combns[1,j], combns[2,j], i, "rds", sep="."))
                }
                if(!is.null(combine)) { # only do this if we need it
                    trainpredictions[,j] = model$predict_newdata(newdata=trf$features)$response
                }
                pairpredictions[,j] = model$predict_newdata(newdata=tsf$features)$response
            }
        } else {
            values = lapply(1:ncol(combns), function(j) {
                a1 = which(data$data[data$train[[i]], data$algos] == combns[1,j])
                a2 = which(data$data[data$train[[i]], data$algos] == combns[2,j])
                trperf = data.frame(target=data$data[data$train[[i]], ][a1, data$performance] - data$data[data$train[[i]], ][a2, data$performance])
                
                f1 = subset(data$data[data$train[[i]], ], data$data[data$train[[i]], data$algos] == combns[1,j])
                f2 = subset(data$data[data$train[[i]], ], data$data[data$train[[i]], data$algos] == combns[2,j])
                f1 = f1[1, data$algorithmFeatures]
                f2 = f2[1, data$algorithmFeatures]
                feature.diff = f1 - f2
                
                tr.features = trf$features[trf$features[[data$algos]] == combns[1,j], ]
                tr.features[[data$algos]] = NULL
                trd = cbind.data.frame(tr.features, feature.diff, trperf)
                
                ts.features = tsf$features[tsf$features[[data$algos]] == combns[1,j], ]
                ts.features[[data$algos]] = NULL
                tsd = cbind.data.frame(ts.features, feature.diff)
                return(list(trd = trd, tsd = tsd))
            })
            
            trf = list()
            tsf = list()
            trf$features = as.data.frame(rbindlist(lapply(values, function(x) { x$trd })))
            tsf$features = as.data.frame(rbindlist(lapply(values, function(x) { x$tsd })))
            
            algo.pairs = rep(1:ncol(combns), each = nrow(values[[1]]$tsd))
            
            task = TaskRegr$new(id="regression", target="target", backend=trf$features) 
            #, fixup.data="quiet")
            model = regressor$train(task = task)
            
            if(!is.na(save.models)) {
                saveRDS(list(model=model, train.data=task, test.data=tsf$features), file = paste(save.models, regressor$id, i, "rds", sep="."))
            }
            pairpredictions = model$predict_newdata(newdata=tsf$features)$response
        }
        
        if(!is.null(combine)) {
            trainBests = data.frame(target=factor(breakBestTies(data, i, pairs=TRUE), levels=data$performance))
            if("weights" %in% combine$properties && use.weights) {
                trw = abs(apply(trp, 1, max) - apply(trp, 1, min))
                data_df = cbind(trainBests, trf$features, data.frame(trainpredictions))
                data_df$weights = trw
                task = TaskClassif$new(id="regression", target="target", backend=data_df)
                #, fixup.data="quiet", check.data=FALSE)
                task$set_col_roles("weights",roles="weight")
            } else {
                task = TaskClassif$new(id="regression", target="target", backend=cbind(trainBests, trf$features, data.frame(trainpredictions)))
                #, fixup.data="quiet", check.data=FALSE)
            }
            
            combinedmodel = combine$train(task = task)
            if(!is.na(save.models)) {
                saveRDS(list(model=combinedmodel, train.data=task, test.data=cbind(tsf$features, data.frame(pairpredictions))), file = paste(save.models, combine$id, "combined", i, "rds", sep="."))
            }
            preds = combinedmodel$predict_newdata(newdata=cbind(tsf$features, data.frame(pairpredictions)))$response
            combinedpredictions = rbind.fill(lapply(1:length(preds), function(j) {
                if(all(is.na(preds[j,drop=F]))) {
                    data.frame(ids[j,,drop=F], algorithm=factor(NA), score=worstScore, iteration=i, row.names = NULL)
                } else {
                    tab = table(preds[j,drop=F])
                    data.frame(ids[j,,drop=F], algorithm=factor(names(tab)), score=as.vector(tab), iteration=i, row.names = NULL)
                }
            }))
        } else {
            if(data$minimize) {
                sign = function(d) { return(-d) }
            } else {
                sign = function(d) { return(d) }
            }
            if(!is.null(data$algorithmFeatures)) {
                pairpredictions = data.frame(target = pairpredictions, algo.pairs = algo.pairs, rep(ids[[data$ids]], ncol(combns)))
                colnames(pairpredictions) = c("target", "algo.pairs", data$ids)
                pairpredictions = convertLongToWide(pairpredictions, timevar="algo.pairs", idvar=data$ids, prefix="target.")
            }
            combinedpredictions = rbind.fill(lapply(1:nrow(pairpredictions), function(j) {
                row = pairpredictions[j,]
                if(is.null(data$algorithmFeatures)) {
                    algos = data$performance
                } else {
                    algos = unique(data$data[[data$algos]])
                }
                performanceSums = sapply(algos, function(p) {
                    sum(unlist(sapply(1:ncol(combns), signPair, p, sign, row, combns)))
                })
                x = sort(performanceSums, decreasing = TRUE)
                if(all(is.na(x))) {
                    data.frame(ids[j,,drop=F], algorithm=factor(NA), score=worstScore, iteration=i, row.names = NULL)
                } else {
                    data.frame(ids[j,,drop=F], algorithm=factor(names(x)), score=unlist(x), iteration=i, row.names = NULL)
                }
            }))
        }
        return(combinedpredictions)
    }, 1:length(data$train), level = "llama.fold"))
    
    if(is.null(data$algorithmFeatures)) {
        fs = pre(data$data[data$features])
    } else {
        fs = pre(data$data[c(data$features, data$algos)]) 
    }
    fp = data$data[data$performance]
    
    fw = abs(apply(fp, 1, max) - apply(fp, 1, min))
    if(is.null(data$algorithmFeatures)) {
        models = lapply(1:ncol(combns), function(i) {
            values = data.frame(target=data$data[[combns[1,i]]] - data$data[[combns[2,i]]])
            task = TaskRegr$new(id="regression", target="target", backend=cbind(values, fs$features))
            #, fixup.data="quiet")
            return(regressor$train(task = task))
        })
    } else {
        values = lapply(1:ncol(combns), function(j) {
            a1 = which(data$data[[data$algos]] == combns[1,j])
            a2 = which(data$data[[data$algos]] == combns[2,j])
            trperf = data.frame(target=data$data[a1, data$performance] - data$data[a2, data$performance])
            
            f1 = subset(data$data, data$data[[data$algos]] == combns[1,j])
            f2 = subset(data$data, data$data[[data$algos]] == combns[2,j])
            f1 = f1[data$algorithmFeatures]
            f2 = f2[data$algorithmFeatures]
            feature.diff = f1 - f2
            
            fs.features = fs$features[fs$features[[data$algos]] == combns[1,j], ]
            fs.features[[data$algos]] = NULL
            fsd = cbind.data.frame(fs.features, feature.diff, trperf)
            return(list(fsd = fsd))
        })
        
        fs = list()
        fs$features = as.data.frame(rbindlist(lapply(values, function(x) { x$fsd })))
        
        algo.pairs = rep(1:ncol(combns), each = nrow(values[[1]]$fsd))
        
        task = TaskRegr$new(id="regression", target="target", backend=fs$features)
        models = regressor$train(task = task)
    }
    
    if(!is.null(combine)) {
        trainpredictions = matrix(nrow=nrow(fs$features), ncol=ncol(combns))
        for(i in 1:ncol(combns)) {
            trainpredictions[,i] = models[[i]]$predict_newdata(newdata=fs$features)$response
        }
        
        if("weights" %in% combine$properties && use.weights) {
            data_df = cbind(totalBests, fs$features, data.frame(trainpredictions))
            data_df$weights = fw
            task = TaskClassif$new(id="regression", target="target", backend = data_df)
            #, fixup.data="quiet", check.data=FALSE)
            task$set_col_roles("weights", roles="weight")
        } else {
            task = TaskClassif$new(id="regression", target="target", backend=cbind(totalBests, fs$features, data.frame(trainpredictions)))
            #, fixup.data="quiet", check.data=FALSE)
        }
        combinedmodel = combine$train(task = task)
    }
    
    predictor = function(x) {
        if(is.null(data$algorithmFeatures)) {
            tsf = pre(x[data$features], fs$meta)
        } else {
            tsf = pre(x[c(data$features, data$algos)], fs$meta)
            values = lapply(1:ncol(combns), function(j) {
                a1 = which(x[[data$algos]] == combns[1,j])
                a2 = which(x[[data$algos]] == combns[2,j])
                trperf = data.frame(target=x[a1, data$performance] - x[a2, data$performance])
                
                f1 = subset(x, x[[data$algos]] == combns[1,j])
                f2 = subset(x, x[[data$algos]] == combns[2,j])
                f1 = f1[data$algorithmFeatures]
                f2 = f2[data$algorithmFeatures]
                feature.diff = f1 - f2
                
                ts.features = tsf$features[tsf$features[[data$algos]] == combns[1,j], ]
                ts.features[[data$algos]] = NULL
                tsd = cbind.data.frame(ts.features, feature.diff)
                return(list(tsd = tsd))
            })
            
            tsf = list()
            tsf$features = as.data.frame(rbindlist(lapply(values, function(x) { x$tsd })))
            
            algo.pairs = rep(1:ncol(combns), each = nrow(values[[1]]$tsd))
        }
        if(is.null(data$algorithmFeatures)) {
            if(length(intersect(colnames(x), data$ids)) > 0) {
                ids = x[data$ids]
            } else {
                ids = data.frame(id = 1:nrow(x)) # don't have IDs, generate them
            }
        } else {
            if(length(intersect(colnames(x), data$ids)) > 0) {
                ids = unique(x[data$ids])
            } else {
                n.unique = length(algo.pairs) / length(unique(algo.pairs))
                ids = data.frame(id = 1:n.unique) # don't have IDs, generate them
            }
        }
        
        if(is.null(data$algorithmFeatures)) {
            pairpredictions = matrix(nrow=nrow(tsf$features), ncol=ncol(combns))
            for(i in 1:ncol(combns)) {
                pairpredictions[,i] = models[[i]]$predict_newdata(newdata=tsf$features)$response
            }
        } else {
            pairpredictions = models$predict_newdata(newdata=tsf$features)$response
        }
        
        if(!is.null(combine)) {
            preds = combinedmodel$predict_newdata(newdata=cbind(tsf$features, data.frame(pairpredictions)))$response
            combinedpredictions = rbind.fill(lapply(1:length(preds), function(j) {
                if(all(is.na(preds[j,drop=F]))) {
                    data.frame(ids[j,,drop=F], algorithm=factor(NA), score=worstScore, iteration=i, row.names = NULL)
                } else {
                    tab = table(preds[j,drop=F])
                    data.frame(ids[j,,drop=F], algorithm=factor(names(tab)), score=as.vector(tab), iteration=1, row.names = NULL)
                }
            }))
        } else {
            if(data$minimize) {
                sign = function(d) { return(-d) }
            } else {
                sign = function(d) { return(d) }
            }
            if(!is.null(data$algorithmFeatures)) {
                ids = unique(ids)
                pairpredictions = data.frame(target = pairpredictions, algo.pairs = algo.pairs, rep(ids[[data$ids]], ncol(combns)))
                colnames(pairpredictions) = c("target", "algo.pairs", data$ids)
                pairpredictions = convertLongToWide(pairpredictions, timevar="algo.pairs", idvar=data$ids, prefix="target.")
            }
            combinedpredictions = rbind.fill(lapply(1:nrow(pairpredictions), function(j) {
                row = pairpredictions[j,]
                if(is.null(data$algorithmFeatures)) {
                    algos = data$performance
                } else {
                    algos = data$algorithmNames 
                }
                performanceSums = sapply(algos, function(p) {
                    sum(unlist(sapply(1:ncol(combns), signPair, p, sign, row, combns)))
                })
                x = sort(performanceSums, decreasing = TRUE)
                if(all(is.na(x))) {
                    data.frame(ids[j,,drop=F], algorithm=factor(NA), score=worstScore, iteration=1, row.names = NULL)
                } else {
                    data.frame(ids[j,,drop=F], algorithm=factor(names(x)), score=unlist(x), iteration=1, row.names = NULL)
                }
            }))
        }
        return(combinedpredictions)
    }
    class(predictor) = "llama.model"
    attr(predictor, "type") = "regressionPairs"
    attr(predictor, "hasPredictions") = FALSE
    attr(predictor, "addCosts") = TRUE
    
    retval = list(predictions=predictions, models=models, predictor=predictor)
    class(retval) = "llama.model"
    attr(retval, "type") = "regressionPairs"
    attr(retval, "hasPredictions") = TRUE
    attr(retval, "addCosts") = TRUE
    
    return(retval)
}
class(regressionPairs) = "llama.modelFunction"

