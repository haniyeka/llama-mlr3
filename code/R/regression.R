regression <-
function(regressor=NULL, data=NULL, pre=function(x, y=NULL) { list(features=x) }, combine=NULL, expand=identity, save.models=NA, use.weights = TRUE) {
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
    worstScore = if(data$minimize) { Inf } else { -Inf }
    
    if(is.null(data$algorithmFeatures)) {
        totalBests = data.frame(target=factor(breakBestTies(data), levels=data$performance))
    } else {
        totalBests = data.frame(target=factor(breakBestTies(data), levels=data$algorithmNames))
    }
    
    predictions = rbind.fill(parallelMap(function(i) {
        if(is.null(data$algorithmFeatures)) {
            trf = pre(data$data[data$train[[i]],][data$features])
            tsf = pre(data$data[data$test[[i]],][data$features], trf$meta)
            trp = data$data[data$train[[i]],][data$performance]
            ids = data$data[data$test[[i]],][data$ids]
        } else {
            trf = pre(data$data[data$train[[i]],][c(data$features, data$algorithmFeatures, data$algos)])
            trf$features = arrange(trf$features, trf$features[[data$algos]])
            trf$features[[data$algos]] = NULL

            tsf = pre(data$data[data$test[[i]],][c(data$features, data$algorithmFeatures, data$algos)], trf$meta)
            tsf$features = arrange(tsf$features, tsf$features[[data$algos]])
            tsf$features[[data$algos]] = NULL

            trp = data$data[data$train[[i]],][c(data$performance, data$algos)]
            trp = arrange(trp, trp[[data$algos]])
            trp[[data$algos]] = NULL

            tsa = data$data[data$test[[i]],][data$algos]
            tsa = arrange(tsa, tsa[[data$algos]])
        		
            ids = data$data[data$test[[i]],][c(data$ids, data$algos)]
            ids = arrange(ids, ids[[data$algos]])
            ids[[data$algos]] = NULL
        }
        
        
        trainpredictions = data.frame(row.names=1:nrow(trf$features))
        performancePredictions = data.frame(row.names=1:nrow(tsf$features))
        if(regressor$predict_type == "se"){
          trainpredictionsSE = data.frame(row.names=1:nrow(trf$features))
          performancePredictionsSE = data.frame(row.names=1:nrow(tsf$features))
        }
        if(is.null(data$algorithmFeatures)) {
            for (j in 1:length(data$performance)) {
                task = TaskRegr$new(id="regression", target="target", backend= cbind(data.frame(target=data$data[data$train[[i]],data$performance[j]]), trf$features))
                model = regressor$train(task = task)
                if(!is.na(save.models)) {
                    saveRDS(list(model=model, train.data=task, test.data=tsf$features), file = paste(save.models, regressor$id, data$performance[[j]], i, "rds", sep="."))
                }
                if(!is.null(combine)) {
                  trainpredictions[,j] = model$predict_newdata(newdata=trf$features)$response
                  if(regressor$predict_type == "se") trainpredictionsSE[,j] = model$predict_newdata(newdata=trf$features)$se
                }
                performancePredictions[,j] = model$predict_newdata(newdata=tsf$features)$response
                if(regressor$predict_type == "se") performancePredictionsSE[,j] = model$predict_newdata(newdata=tsf$features)$se
            }
            colnames(performancePredictions) = data$performance
            if(regressor$predict_type == "se") colnames(performancePredictionsSE) = data$performance
        } else {
            target = data$data[data$train[[i]], c(data$performance, data$algos)]
            target = arrange(target, target[[data$algos]])
            target[[data$algos]] = NULL
            target = unname(unlist(target))
            task = TaskRegr$new(id="regression", target="target", backend=cbind(data.frame(target=target, trf$features)))
            model = regressor$train(task = task)
            if(!is.na(save.models)) {
                saveRDS(list(model=model, train.data=task, test.data=tsf$features), file = paste(save.models, regressor$id, i, "rds", sep="."))
            }
            performancePredictions[[data$performance]] = model$predict_newdata(newdata=tsf$features)$response
            if(regressor$predict.type == "se") performancePredictionsSE[[data$performance]] = model$predict_newdata(newdata=tsf$features)$se
        }
        
        if(!is.null(combine)) {
            colnames(trainpredictions) = data$performance
            if(regressor$predict_type == "se") colnames(trainpredictionsSE) = data$performance
            trainBests = data.frame(target=factor(breakBestTies(data, i), levels=data$performance))
            if("weights" %in% combine$properties && use.weights) {
                trw = abs(apply(trp, 1, max) - apply(trp, 1, min))
                data_df = cbind(trainBests, trf$features, data.frame(expand(trainpredictions)))
                data_df$weights = trw
                task = TaskClassif$new(id="regression", target="target", backend = data_df)
                task$set_col_roles("weights", roles = "weight")
            } else {
                task = TaskClassif$new(id="regression", target="target", backend = cbind(trainBests, trf$features, data.frame(expand(trainpredictions))))
            }
            combinedmodel = combine$train(task = task)
            if(!is.na(save.models)) {
                saveRDS(list(model=combinedmodel, train.data=task, test.data=cbind(tsf$features, data.frame(expand(performancePredictions)))), file = paste(save.models, combine$id, "combined", i, "rds", sep="."))
            }
            preds = combinedmodel$predict_newdata(newdata=cbind(tsf$features, data.frame(expand(performancePredictions))))$response
            if(regressor$predict_type == "se") predsSE = combinedmodel$predict_newdata(newdata=cbind(tsf$features, data.frame(expand(performancePredictions))))$se
            combinedpredictions = rbind.fill(lapply(1:length(preds), function(j) {
                if(all(is.na(preds[j,drop=F]))) {
                    data.frame(ids[j,,drop=F], algorithm=factor(NA), score=worstScore, iteration=i, row.names = NULL)
                } else {
                    tab = as.table(sort(table(preds[j,drop=F]), decreasing=T))
                    data.frame(ids[j,,drop=F], algorithm=factor(names(tab)), score=as.vector(tab), iteration=i, row.names = NULL)
                }
            }))
        } else {
            if(!is.null(data$algorithmFeatures)) {
                performancePredictions[[data$algos]] = tsa[[data$algos]]
                performancePredictions[[data$ids]] = ids[[data$ids]]
                performancePredictions = convertLongToWide(performancePredictions, timevar=data$algos, idvar=data$ids, prefix=paste(data$performance,".",sep=""), remove.id = FALSE)
                ids = performancePredictions[data$ids]
                performancePredictions[data$ids] = NULL
                
                if(regressor$predict_type == "se"){
                  performancePredictionsSE[[data$algos]] = tsa[[data$algos]]
                  performancePredictionsSE[[data$ids]] = ids[[data$ids]]
                  performancePredictionsSE = convertLongToWide(performancePredictionsSE, timevar=data$algos, idvar=data$ids, prefix=paste(data$performance,".",sep=""), remove.id = FALSE)
                  ids = performancePredictionsSE[data$ids]
                  performancePredictionsSE[data$ids] = NULL
                }
            }
            
            combinedpredictions = rbind.fill(lapply(1:nrow(performancePredictions), function(j) {
                if(all(is.na(performancePredictions[j,]))) {
                    data.frame(ids[j,,drop=F], algorithm=factor(NA), score=worstScore, iteration=i, row.names = NULL)
                } else {
                    x = sort(unlist(performancePredictions[j,,drop=F]), decreasing = !data$minimize)
                    if(regressor$predict_type == "se"){
                      xSE = unlist(performancePredictionsSE[j,,drop=F])[order(unlist(performancePredictions[j,,drop=F]), decreasing = !data$minimize)]
                      data.frame(ids[j,,drop=F], algorithm=factor(names(x)), score=unlist(x), se=unlist(xSE), iteration=i, row.names = NULL)
                    } else{
                      data.frame(ids[j,,drop=F], algorithm=factor(names(x)), score=unlist(x), iteration=i, row.names = NULL)
                    }
                }
            }))
        }
        return(combinedpredictions)
    }, 1:length(data$train), level = "llama.fold"))
    
    if(is.null(data$algorithmFeatures)) {
        fs = pre(data$data[data$features])
        fp = data$data[data$performance]
    } else {
        fs = pre(data$data[c(data$features, data$algorithmFeatures, data$algos)])
        fs$features = arrange(fs$features, fs$features[[data$algos]])
        fs$features[[data$algos]] = NULL

        fp = data$data[c(data$performance, data$algos)]
        fp = arrange(fp, fp[[data$algos]])
        fp[[data$algos]] = NULL
    }
    
    fw = abs(apply(fp, 1, max) - apply(fp, 1, min))
    if(is.null(data$algorithmFeatures)) {
        models = lapply(1:length(data$performance), function(i) {
            task = TaskRegr$new(id="regression", target="target", backend=cbind(data.frame(target=data$data[[data$performance[i]]]), fs$features))
            return(regressor$train(task = task))
        })
    } else {
        target = data$data[c(data$performance, data$algos)]
        target = arrange(target, target[[data$algos]])
        target[[data$algos]] = NULL
        target = unname(unlist(target))
        task = TaskRegr$new(id="regression", target="target", backend=cbind(data.frame(target=target, fs$features)))
        models = regressor$train(task = task)
    }
    
    if(!is.null(combine)) {
        trainpredictions = data.frame(row.names=1:nrow(fs$features))
        if(regressor$predict_type == "se") trainpredictionsSE = data.frame(row.names=1:nrow(fs$features))
        if(is.null(data$algorithmFeatures)) {
            for (i in 1:length(data$performance)) {
                trainpredictions[,i] = models[[i]]$predict_newdata(newdata=fs$features)$response
                if(regressor$predict_type == "se") trainpredictionsSE[,i] = models[[i]]$predict_newdata(newdata=fs$features)$se
            }
            colnames(trainpredictions) = data$performance
            if(regressor$predict_type == "se") colnames(trainpredictionsSE) = data$performance
        } else {
            trainpredictions[[data$performance]] = models$predict_newdata(newdata=fs$features)$response
            if(regressor$predict_type == "se") trainpredictionsSE[[data$performance]] = models$predict_newdata(newdata=fs$features)$se
        }
        
        if("weights" %in% combine$properties && use.weights) {
            data_df = cbind(totalBests, fs$features, data.frame(expand(trainpredictions)))
            data_df$weights = fw
            task = TaskClassif$new(id="regression", target="target", backend = data_df)
            task$set_col_roles("weights", roles = "weight")
        } else {
            task = TaskClassif$new(id="regression", target="target", backend = cbind(totalBests, fs$features, data.frame(expand(trainpredictions))))
        }
        combinedmodel = combine$train(task = task)
    }
    
    predictor = function(x) {
        if(is.null(data$algorithmFeatures)) {
            tsf = pre(x[data$features], fs$meta)
        } else {
            tsf = pre(x[c(data$features, data$algorithmFeatures, data$algos)], fs$meta)
            tsf$features = arrange(tsf$features, tsf$features[[data$algos]])
            tsf$features[[data$algos]] = NULL

            tsa = x[data$algos]
            tsa = arrange(tsa, tsa[[data$algos]])
        }
        if(length(intersect(colnames(x), data$ids)) > 0) {
            if(is.null(data$algorithmFeatures)) {
              ids = x[data$ids]
            } else {
              ids = x[c(data$ids, data$algos)]
              ids = arrange(ids, ids[[data$algos]])
              ids[[data$algos]] = NULL
						}
				} else {
            # don't have IDs, generate them
            if(is.null(data$algorithmFeatures)) {
                ids = data.frame(id = 1:nrow(x))  
            } else {
                n = nrow(x) / length(unique(x[[data$algos]]))
                ids = data.frame(id = rep.int(1:n, length(unique(x[[data$algos]]))))
            }
        }
        performancePredictions = data.frame(row.names=1:nrow(tsf$features))
        if(regressor$predict_type == "se") performancePredictionsSE = data.frame(row.names=1:nrow(tsf$features))
        if(is.null(data$algorithmFeatures)) {
            for (i in 1:length(data$performance)) {
                performancePredictions[,i] = models[[i]]$predict_newdata(newdata=tsf$features)$response
                if(regressor$predict_type == "se") performancePredictionsSE[,i] = models[[i]]$predict_newdata(newdata=tsf$features)$response
            }
            colnames(performancePredictions) = data$performance
            if(regressor$predict_type == "se") { colnames(performancePredictionsSE) = data$performance }
        } else {
            performancePredictions[[data$performance]] = models$predict_newdata(newdata=tsf$features)$response
            if(regressor$predict_type == "se") performancePredictionsSE[[data$performance]] = models$predict_newdata(newdata=tsf$features)$se
        }
        
        if(!is.null(combine)) {
            preds = combinedmodel$predict_newdata(newdata=cbind(tsf$features, data.frame(expand(performancePredictions))))$response
            if(regressor$predict_type == "se") predsSE = combinedmodel$predict_newdata(newdata=cbind(tsf$features, data.frame(expand(performancePredictions))))$se
            combinedpredictions = rbind.fill(lapply(1:length(preds), function(j) {
                if(all(is.na(preds[j,drop=F]))) {
                    data.frame(ids[j,,drop=F], algorithm=factor(NA), score=worstScore, iteration=1, row.names = NULL)
                } else {
                    tab = as.table(sort(table(preds[j,drop=F]), decreasing=T))
                    data.frame(ids[j,,drop=F], algorithm=factor(names(tab)), score=as.vector(tab), iteration=1, row.names = NULL)
                }
            }))
        } else {
            if(!is.null(data$algorithmFeatures)) {
                performancePredictions[[data$algos]] = tsa[[data$algos]]
                performancePredictions[[data$ids]] = ids[[data$ids]]
                performancePredictions = convertLongToWide(performancePredictions, timevar=data$algos, idvar=data$ids, prefix=paste(data$performance,".",sep=""), remove.id = FALSE)
                ids = performancePredictions[data$ids]
                performancePredictions[data$ids] = NULL
            }
            
            combinedpredictions = rbind.fill(lapply(1:nrow(performancePredictions), function(j) {
                if(all(is.na(performancePredictions[j,]))) {
                    data.frame(ids[j,,drop=F], algorithm=factor(NA), score=worstScore, iteration=1, row.names = NULL)
                } else {
                    x = sort(unlist(performancePredictions[j,]), decreasing = !data$minimize)
                    if(regressor$predict_type == "se"){
                      xSE = unlist(performancePredictionsSE[j,,drop=F])[order(unlist(performancePredictions[j,,drop=F]), decreasing = !data$minimize)]
                      data.frame(ids[j,,drop=F], algorithm=factor(names(x)), score=unlist(x), se=unlist(xSE), iteration=1, row.names = NULL)
                    } else{
                      data.frame(ids[j,,drop=F], algorithm=factor(names(x)), score=unlist(x), iteration=1, row.names = NULL)
                    }
                }
            }))
        }
        return(combinedpredictions)
    }
    class(predictor) = "llama.model"
    attr(predictor, "type") = "regression"
    attr(predictor, "hasPredictions") = FALSE
    attr(predictor, "addCosts") = TRUE
    
    retval = list(predictions=predictions, models=models, predictor=predictor)
    class(retval) = "llama.model"
    attr(retval, "type") = "regression"
    attr(retval, "hasPredictions") = TRUE
    attr(retval, "addCosts") = TRUE
    
    return(retval)
}
class(regression) = "llama.modelFunction"
