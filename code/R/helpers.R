print.llama.data <-
function(x, ...) {
  if(is.null(x$algorithmFeatures)) {
    cat(
      nrow(x$data), " instances\n",
      length(x$performance), " algorithms\n",
      "ID columns: ", paste(x$ids, collapse=", "), "\n",
      "Instance Features: ", paste(x$features, collapse=", "), "\n",
      "Performances: ", paste(x$performance, collapse=", "), "\n",
      "Successes: ", paste(x$success, collapse=", "), "\n",
      "Cost groups: ", printList(x$costGroups), "\n",
      "Extra: ", paste(x$extra, collapse=", "), "\n",
      "Minimize: ", x$minimize, "\n",
      "Has splits: ", attr(x, "hasSplits"), "\n",
      sep = "")
  } else {
    cat(
      nrow(unique(x$data[x$ids])), " instances\n",
			length(x$algorithmNames), " algorithms\n",
      "ID columns: ", paste(x$ids, collapse=", "), "\n",
      "Algorithm columns: ", paste(x$algos, collapse=", "), "\n",
      "Instance Features: ", paste(x$features, collapse=", "), "\n",
      "Algorithm Features: ", paste(x$algorithmFeatures, collapse=", "), "\n",
      "Performances: ", paste(x$performance, collapse=", "), "\n",
      "Successes: ", paste(x$success, collapse=", "), "\n",
      "Cost groups: ", printList(x$costGroups), "\n",
      "Extra: ", paste(x$extra, collapse=", "), "\n",
      "Minimize: ", x$minimize, "\n",
      "Has splits: ", attr(x, "hasSplits"), "\n",
      sep = "")
  }
}

print.llama.model <-
function(x, ...) {
    cat(
      "Type: ", attr(x, "type"), "\n",
      "Has predictions: ", attr(x, "hasPredictions"), "\n",
      "Add costs: ", attr(x, "addCosts"), "\n",
      "Tuned: ", (length(x$parvals) > 0), "\n",
      sep = "")
}

printList <-
function(l) {
    paste(sapply(names(l), function(x) {
        paste(x, " = [", paste(l[[x]], collapse=", "), "]", sep="")
    }), collapse="")
}

skip.expensive <-
function() {
    cond = structure(list(message = "Skipping expensive run."), class = c("skip", "condition"))
    if(Sys.getenv("RUN_EXPENSIVE") != "true") stop(cond)
}

