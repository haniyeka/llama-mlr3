print.llama.data <-
function(data) {
    cat(
      nrow(data$data), " instances\n",
      length(data$performance), " algorithms\n",
      "ID columns: ", collapse(data$ids, sep=", "), "\n",
      "Features: ", collapse(data$features, sep=", "), "\n",
      "Performances: ", collapse(data$performance, sep=", "), "\n",
      "Successes: ", collapse(data$success, sep=", "), "\n",
      "Cost groups: ", printList(data$costGroups), "\n",
      "Extra: ", collapse(data$extra, sep=", "), "\n",
      "Minimize: ", data$minimize, "\n",
      "Has splits: ", attr(data, "hasSplits"), "\n",
      sep = "")
}

print.llama.model <-
function(model) {
    cat(
      "Type: ", attr(model, "type"), "\n",
      "Has predictions: ", attr(model, "hasPredictions"), "\n",
      "Add costs: ", attr(model, "addCosts"), "\n",
      sep = "")
}

printList <-
function(l) {
    collapse(sapply(names(l), function(x) {
        paste(x, " = [", collapse(l[[x]], sep=", "), "]", sep="")
    }))
}
