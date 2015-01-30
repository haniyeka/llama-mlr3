print.llama.data <-
function(x, ...) {
    cat(
      nrow(x$data), " instances\n",
      length(x$performance), " algorithms\n",
      "ID columns: ", collapse(x$ids, sep=", "), "\n",
      "Features: ", collapse(x$features, sep=", "), "\n",
      "Performances: ", collapse(x$performance, sep=", "), "\n",
      "Successes: ", collapse(x$success, sep=", "), "\n",
      "Cost groups: ", printList(x$costGroups), "\n",
      "Extra: ", collapse(x$extra, sep=", "), "\n",
      "Minimize: ", x$minimize, "\n",
      "Has splits: ", attr(x, "hasSplits"), "\n",
      sep = "")
}

print.llama.model <-
function(x, ...) {
    cat(
      "Type: ", attr(x, "type"), "\n",
      "Has predictions: ", attr(x, "hasPredictions"), "\n",
      "Add costs: ", attr(x, "addCosts"), "\n",
      sep = "")
}

printList <-
function(l) {
    collapse(sapply(names(l), function(x) {
        paste(x, " = [", collapse(l[[x]], sep=", "), "]", sep="")
    }))
}
