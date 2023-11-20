classify <-
function(classifier=NULL, data=NULL, pre=function(x, y=NULL) { list(features=x) }, save.models=NA, use.weights = TRUE) {
    if(!testClass(classifier, "Learner") && !testList(classifier, types="Learner")) {
        stop("Need classifier or list of classifiers!")
    }
    assertClass(data, "llama.data")
    hs = attr(data, "hasSplits")
    if(is.null(hs) || hs != TRUE) {
        stop("Need data with train/test split!")
    }
    if(testClass(classifier, "Learner")) { classifier = list(classifier) }
    combinator = "majority"
    if(!is.null(classifier$.combine)) {
        combinator = classifier$.combine
        classifier = classifier[-which(names(classifier) == ".combine")]
    }

    totalBests = data.frame(target=factor(breakBestTies(data), levels=data$performance))

    predictions = rbind.fill(parallelMap(function(i) {
        trf = pre(data$data[data$train[[i]],][data$features])
        tsf = pre(data$data[data$test[[i]],][data$features], trf$meta)
        ids = data$data[data$test[[i]],][data$ids]
        trp = data$data[data$train[[i]],][data$performance]
        trw = abs(apply(trp, 1, max) - apply(trp, 1, min))

        trainpredictions = list()
        ensemblepredictions = list()
        
        trainBests = data.frame(target=factor(breakBestTies(data, i), levels=data$performance))
        for(j in 1:length(classifier)) {
          if("weights" %in% classifier[[j]]$properties && use.weights) {
            data_df = data.frame(trainBests, trf$features)
            data_df$weights = trw
            task = TaskClassif$new(id = "classify", backend = data_df, target = "target")
            task$set_col_roles("weights", roles = "weight")
          } else {
            task = TaskClassif$new(id = "classify", backend = data.frame(trainBests, trf$features), target = "target")
          }
          if(length(unique(trainBests$target)) == 1) {
            # one-class problem
            model = constantClassifier$train(task)
          } else {
            model = classifier[[j]]$train(task)
          }
          if(!is.na(save.models)) {
            saveRDS(list(model=model, train.data=task, test.data=tsf$features), file = paste(save.models, classifier[[j]]$id, i, "rds", sep="."))
          }
          if(inherits(combinator, "Learner")) { # only do this if we need it
            preds = model$predict_newdata(newdata=trf$features)
            trainpredictions[[j]] = if("prob" %in% preds$predict_types) {
              preds$prob
            } else {
              tmp = preds$response
              rbind.fill(lapply(tmp, function(x) data.frame(t(setNames(as.numeric(x == levels(tmp)), levels(tmp))))))
            }
          }
          preds = model$predict_newdata(newdata=tsf$features)
          ensemblepredictions[[j]] = if("prob" %in% preds$predict_types) {
            preds$prob
          } else {
            tmp = preds$response
            rbind.fill(lapply(tmp, function(x) data.frame(t(setNames(as.numeric(x == levels(tmp)), levels(tmp))))))
          }
        }
        if(inherits(combinator, "Learner")) {
          if("weights" %in% classifier[[j]]$properties && use.weights) {
            data_df = data.frame(trainBests, trf$features, trainpredictions)
            data_df$weights = trw
            task = TaskClassif$new(id = "classify", backend = data_df, target = "target")
            task$set_col_roles("weights", roles = "weight")
          } else {
            task = TaskClassif$new(id = "classify", backend = data.frame(trainBests, trf$features, trainpredictions), target = "target")
          }
          if(length(unique(trainBests$target)) == 1) {
            # one-class problem
            combinedmodel = constantClassifier$train(task = task)
          } else {
            combinedmodel = combinator$train(task = task)
          }
          if(!is.na(save.models)) {
            saveRDS(list(model=combinedmodel, train.data=task, test.data=data.frame(tsf$features, ensemblepredictions)), file = paste(save.models, combinator$id, "combined", i, "rds", sep="."))
          }
          preds = combinedmodel$predict_newdata(newdata=data.frame(tsf$features, ensemblepredictions))
          if("prob" %in% preds$predict_types) {
            preds = preds$prob
          } else {
            preds = preds$response
            preds = rbind.fill(lapply(preds, function(x) data.frame(t(setNames(as.numeric(x == levels(preds)), levels(preds))))))
          }
          combinedpredictions = rbind.fill(lapply(1:nrow(preds), function(j) {
            ss = preds[j,,drop=F]
            ord = order(unlist(ss), decreasing = TRUE)
            data.frame(ids[j,,drop=F], algorithm=factor(colnames(ss)[ord]), score=as.numeric(ss)[ord], iteration=i, row.names = NULL)
          }))
        } else {
          merged = Reduce('+', ensemblepredictions)
          combinedpredictions = rbind.fill(lapply(1:nrow(merged), function(j) {
            ord = order(unlist(merged[j,]), decreasing = TRUE)
            data.frame(ids[j,,drop=F], algorithm=factor(colnames(merged)[ord]), score=as.numeric(merged[j,])[ord], iteration=i, row.names = NULL)
          }))
        }
        return(combinedpredictions)
    }, 1:length(data$train), level = "llama.fold"))
    
    fs = pre(data$data[data$features])
    fp = data$data[data$performance]
    fw = abs(apply(fp, 1, max) - apply(fp, 1, min))
    models = lapply(1:length(classifier), function(i) {
      if("weights" %in% classifier[[i]]$properties && use.weights) {
        data_df = data.frame(totalBests, fs$features)
        data_df$weights = fw
        task = TaskClassif$new(id = "classify", backend = data_df, target = "target")
        task$set_col_roles("weights", roles = "weight")
      } else {            
        task = TaskClassif$new(id = "classify", backend = data.frame(totalBests, fs$features), target = "target")
      }
      if(length(unique(totalBests$target)) == 1) {
        # one-class problem
        model = constantClassifier$train(task = task)
      } else {
        model = classifier[[i]]$train(task = task)
      }
      return(model)
    })
    if(inherits(combinator, "Learner")) {
      trainpredictions = list()
      for(i in 1:length(classifier)) {
        preds = models[[i]]$predict_newdata(newdata=fs$features)
        trainpredictions[[i]] = if("prob" %in% preds$predict_types) {
          preds$prob
        } else {
          tmp = preds$response
          rbind.fill(lapply(tmp, function(x) data.frame(t(setNames(as.numeric(x == levels(tmp)), levels(tmp))))))
        }
      }
      if("weights" %in% combinator$properties && use.weights) {
        data_df = data.frame(totalBests, fs$features, trainpredictions)
        data_df$weights = fw
        task = TaskClassif$new(id = "classify", backend = data_df, target = "target")
        task$set_col_roles("weights", roles = "weight")
      } else {
        task = TaskClassif$new(id="classify", target="target", backend = data.frame(totalBests, fs$features, trainpredictions))
      }
      if(length(unique(totalBests$target)) == 1) {
        # one-class problem
        combinedmodel = constantClassifier$train(task = task)
      } else {
        combinedmodel = combinator$train(task = task)
      }
    }
    
    predictor = function(x) {
      tsf = pre(x[data$features], fs$meta)
      if(length(intersect(colnames(x), data$ids)) > 0) {
        ids = x[data$ids]
      } else {
        ids = data.frame(id = 1:nrow(x)) # don't have IDs, generate them
      }
      ensemblepredictions = list()
      for(i in 1:length(classifier)) {
        preds = models[[i]]$predict_newdata(newdata=tsf$features)
        ensemblepredictions[[i]] = if("prob" %in% preds$predict_types) {
          tmp = preds$prob
        } else {
          tmp = preds$response
          rbind.fill(lapply(tmp, function(x) data.frame(t(setNames(as.numeric(x == levels(tmp)), levels(tmp))))))
        }
      }
      if(inherits(combinator, "Learner")) {
        preds = combinedmodel$predict_newdata(newdata=data.frame(tsf$features, ensemblepredictions))
        if("prob" %in% preds$predict_types) {
          preds = preds$prob
        } else {
          preds = preds$response
          preds = rbind.fill(lapply(preds, function(x) data.frame(t(setNames(as.numeric(x == levels(preds)), levels(preds))))))
        }
        combinedpredictions = rbind.fill(lapply(1:nrow(preds), function(j) {
          ss = preds[j,,drop=F]
          ord = order(unlist(ss), decreasing = TRUE)
          data.frame(ids[j,,drop=F], algorithm=factor(colnames(ss)[ord]), score=as.numeric(ss)[ord], iteration=i, row.names = NULL)
        }))
      } else {
        merged = Reduce('+', ensemblepredictions)
        combinedpredictions = rbind.fill(lapply(1:nrow(merged), function(j) {
          ord = order(unlist(merged[j,]), decreasing = TRUE)
          data.frame(ids[j,,drop=F], algorithm=factor(colnames(merged)[ord]), score=as.numeric(merged[j,])[ord], iteration=i, row.names = NULL)
        }))
      }
      return(combinedpredictions)
    }
    class(predictor) = "llama.model"
    attr(predictor, "type") = "classify"
    attr(predictor, "hasPredictions") = FALSE
    attr(predictor, "addCosts") = TRUE
    
    retval = list(predictions=predictions, models=models, predictor=predictor)
    class(retval) = "llama.model"
    attr(retval, "type") = "classify"
    attr(retval, "hasPredictions") = TRUE
    attr(retval, "addCosts") = TRUE
    
    return(retval)
}
class(classify) = "llama.modelFunction"
