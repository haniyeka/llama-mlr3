classifyPairs <-
function(classifier=NULL, data=NULL, pre=function(x, y=NULL) { list(features=x) }, combine=NULL, save.models=NA, use.weights = TRUE) {
    if(!testClass(classifier, "Learner")) {
        stop("Need classifier!")
    }
    assertClass(data, "llama.data")
    hs = attr(data, "hasSplits")
    if(is.null(hs) || hs != TRUE) {
        stop("Need data with train/test split!")
    }

    totalBests = data.frame(target=factor(breakBestTies(data), levels=data$performance))
    combns = combn(data$performance, 2)
    predictions = rbind.fill(parallelMap(function(i) {
        trf = pre(data$data[data$train[[i]],][data$features])
        tsf = pre(data$data[data$test[[i]],][data$features], trf$meta)
        ids = data$data[data$test[[i]],][data$ids]
        trp = data$data[data$train[[i]],][data$performance]

        trainpredictions = list()
        pairpredictions = list()
        for (j in 1:ncol(combns)) {
            if(data$minimize) {
                cmp = function(x, y) {
                    sapply(data$data[data$train[[i]],][x] < data$data[data$train[[i]],][y], function(z) { if(z) { x } else { y } })
                }
            } else {
                cmp = function(x, y) {
                    sapply(data$data[data$train[[i]],][x] > data$data[data$train[[i]],][y], function(z) { if(z) { x } else { y } })
                }
            }
            labels = data.frame(target=factor(cmp(combns[1,j], combns[2,j])))
            levs = combns[,j]
            if("weights" %in% classifier$properties && use.weights) {
                trw = abs(data$data[data$train[[i]],combns[1,j]] - data$data[data$train[[i]],combns[2,j]])
                data_df = data.frame(labels, trf$features)
                data_df$weights = trw
                task = TaskClassif$new(id="classifyPairs", target="target", backend = data_df)
                #, fixup.data="quiet", check.data=FALSE
                task$set_col_roles("weights", roles = "weight")
            } else {
                task = TaskClassif$new(id="classifyPairs", target="target", backend = data.frame(labels, trf$features))
                #, fixup.data="quiet", check.data=FALSE
            }
            if(length(unique(labels$target)) == 1) {
                # one-class problem
                model = constantClassifier$train(task = task)
            } else {
                model = classifier$train(task = task)
            }
            if(!is.na(save.models)) {
                saveRDS(list(model=model, train.data=task, test.data=tsf$features), file = paste(save.models, classifier$id, combns[1,j], combns[2,j], i, "rds", sep="."))
            }
            if(!is.null(combine)) { # only do this if we need it
                preds = model$predict_newdata(newdata=trf$features)
                trainpredictions[[j]] = if("prob" %in% preds$predict_types) {
                    preds$prob 
                } else {
                    tmp = preds$response
                    rbind.fill(lapply(tmp, function(x) data.frame(t(setNames(as.numeric(x == levs), levs)))))
                }
            }
            preds = model$predict_newdata(newdata=tsf$features)
            pairpredictions[[j]] = if("prob" %in% preds$predict_types) {
                preds$prob 
            } else {
                tmp = preds$response
                rbind.fill(lapply(tmp, function(x) data.frame(t(setNames(as.numeric(x == levs), levs)))))
            }
        }

        if(!is.null(combine)) {
            trainBests = data.frame(target=factor(breakBestTies(data, i), levels=data$performance))
            if("weights" %in% combine$properties && use.weights) {
                trw = abs(apply(trp, 1, max) - apply(trp, 1, min))
                data_df = data.frame(trainBests, trf$features, trainpredictions)
                data_df$weights = trw
                task = TaskClassif$new(id="classifyPairs", target="target", backend = data_df)
                                       #, fixup.data="quiet", check.data=FALSE)
                task$set_col_roles("weights", roles = "weight")
            } else {
                task = TaskClassif$new(id="classifyPairs", target="target", backend = data.frame(trainBests, trf$features, trainpredictions))
                                       #, fixup.data="quiet", check.data=FALSE)
            }
            if(length(unique(trainBests$target)) == 1) {
                # one-class problem
                combinedmodel = constantClassifier$train(task = task)
            } else {
                combinedmodel = combine$train(task = task)
            }
            if(!is.na(save.models)) {
                saveRDS(list(model=combinedmodel, train.data=task, test.data=data.frame(tsf$features, pairpredictions)), file = paste(save.models, combine$id, "combined", i, "rds", sep="."))
            }
            preds = combinedmodel$predict_newdata(newdata=data.frame(tsf$features, pairpredictions))
            if("prob" %in% preds$predict_types) {
                preds = preds$prob
            } else {
                preds = preds$response 
                preds = rbind.fill(lapply(preds, function(x) data.frame(t(setNames(as.numeric(x == levels(preds)), levels(preds))))))
            }
            combinedpredictions = rbind.fill(lapply(1:nrow(preds), function(j) {
                ss = preds[j,,drop=F]
                ord = order(unlist(ss), decreasing = TRUE)
                data.frame(ids[j,,drop=F], algorithm=factor(names(ss)[ord]), score=as.numeric(ss)[ord], iteration=i, row.names = NULL)
            }))
        } else {
            tmp = cbind(tmp.names = unlist(lapply(pairpredictions, rownames)), rbind.fill(pairpredictions))
            merged = ddply(tmp, "tmp.names", function(x) colSums(x[,-1], na.rm = TRUE))[-1]
            combinedpredictions = rbind.fill(lapply(1:nrow(merged), function(j) {
                ord = order(unlist(merged[j,]), decreasing = TRUE)
                data.frame(ids[j,,drop=F], algorithm=factor(names(merged)[ord]), score=as.numeric(merged[j,])[ord], iteration=i, row.names = NULL)
            }))
        }
        return(combinedpredictions)
    }, 1:length(data$train), level = "llama.fold"))

    fs = pre(data$data[data$features])
    fp = data$data[data$performance]
    fw = abs(apply(fp, 1, max) - apply(fp, 1, min))
    models = lapply(1:ncol(combns), function(i) {
        if(data$minimize) {
            cmp = function(x, y) {
                sapply(data$data[[x]] < data$data[[y]], function(z) { if(z) { x } else { y } })
            }
        } else {
            cmp = function(x, y) {
                sapply(data$data[[x]] > data$data[[y]], function(z) { if(z) { x } else { y } })
            }
        }
        labels = data.frame(target=factor(cmp(combns[1,i], combns[2,i])))
        if("weights" %in% classifier$properties && use.weights) {
            data_df = data.frame(labels, fs$features)
            data_df$weights = abs(data$data[[combns[1,i]]] - data$data[[combns[2,i]]])
            task = TaskClassif$new(id="classifyPairs", target="target", backend = data_df)
            #, fixup.data="quiet", check.data=FALSE)
            task$set_col_roles("weights", roles = "weight")
        } else {
            task = TaskClassif$new(id="classifyPairs", target="target", backend = data.frame(labels, fs$features))
            #, fixup.data="quiet", check.data=FALSE)
        }
        if(length(unique(labels$target)) == 1) {
            # one-class problem
            model = constantClassifier$train(task = task)
        } else {
            model = classifier$train(task = task)
        }
        return(model)
    })
    if(!is.null(combine)) {
        trainpredictions = list()
        for(i in 1:ncol(combns)) {
            levs = combns[,i]
            preds = models[[i]]$predict_newdata(newdata=fs$features)
            trainpredictions[[i]] = if("prob" %in% preds$predict_types) {
                preds$prob
            } else {
                tmp = preds$response 
                rbind.fill(lapply(tmp, function(x) data.frame(t(setNames(as.numeric(x == levs), levs)))))
            }
        }
        if("weights" %in% combine$properties && use.weights) {
            data_df = data.frame(totalBests, fs$features, trainpredictions)
            data_df$weights = fw
            task = TaskClassif$new(id="classifyPairs", target="target", backend = data_df)
            #, fixup.data="quiet", check.data=FALSE)
            task$set_col_roles("weights", roles = "weight")
        } else {
            task = TaskClassif$new(id="classifyPairs", target="target", backend = data.frame(totalBests, fs$features, trainpredictions))
            #, fixup.data="quiet", check.data=FALSE)
        }
        if(length(unique(totalBests$target)) == 1) {
            # one-class problem
            combinedmodel = constantClassifier$train(task = task)
        } else {
            combinedmodel = combine$train(task = task)
        }
    }

    predictor = function(x) {
        tsf = pre(x[data$features], fs$meta)
        if(length(intersect(colnames(x), data$ids)) > 0) {
            ids = x[data$ids]
        } else {
            ids = data.frame(id = 1:nrow(x)) # don't have IDs, generate them
        }
        pairpredictions = list()
        for(i in 1:ncol(combns)) {
            levs = combns[,i]
            preds = models[[i]]$predict_newdata(newdata=tsf$features)
            pairpredictions[[i]] = if("prob" %in% preds$predict_types) {
                preds$prob
            } else {
                tmp = preds$response
                rbind.fill(lapply(tmp, function(x) data.frame(t(setNames(as.numeric(x == levs), levs)))))
            }
        }
        if(!is.null(combine)) {
            preds = combinedmodel$predict_newdata(newdata=data.frame(tsf$features, pairpredictions))
            if("prob" %in% preds$predict_types) {
                preds = preds$prob
            } else {
                preds = preds$response
                preds = rbind.fill(lapply(preds, function(x) data.frame(t(setNames(as.numeric(x == levels(preds)), levels(preds))))))
            }
            combinedpredictions = rbind.fill(lapply(1:nrow(preds), function(j) {
                ss = preds[j,,drop=F]
                ord = order(unlist(ss), decreasing = TRUE)
                data.frame(ids[j,,drop=F], algorithm=factor(names(ss)[ord]), score=as.numeric(ss)[ord], iteration=i, row.names = NULL)
            }))
        } else {
            tmp = cbind(tmp.names = unlist(lapply(pairpredictions, rownames)), rbind.fill(pairpredictions))
            merged = ddply(tmp, "tmp.names", function(x) colSums(x[,-1], na.rm = TRUE))[-1]
            combinedpredictions = rbind.fill(lapply(1:nrow(merged), function(j) {
                ord = order(unlist(merged[j,]), decreasing = TRUE)
                data.frame(ids[j,,drop=F], algorithm=factor(names(merged)[ord]), score=as.numeric(merged[j,])[ord], iteration=i, row.names = NULL)
            }))
        }
        return(combinedpredictions)
    }
    class(predictor) = "llama.model"
    attr(predictor, "type") = "classifyPairs"
    attr(predictor, "hasPredictions") = FALSE
    attr(predictor, "addCosts") = TRUE

    retval = list(predictions=predictions, models=models, predictor=predictor)
    class(retval) = "llama.model"
    attr(retval, "type") = "classifyPairs"
    attr(retval, "hasPredictions") = TRUE
    attr(retval, "addCosts") = TRUE

    return(retval)
}
class(classifyPairs) = "llama.modelFunction"
