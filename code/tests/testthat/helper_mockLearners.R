testclassifier = lrn("classif.featureless")
testclassifier$properties = c("numerics", "factors", "oneclass", "twoclass", "multiclass", "prob", "weights")
testclassifier$id = "classif.test"
testclassifier$packages = "llama"

tuningtestclassifier = lrn("classif.ranger")

natestclassifier = lrn("classif.featureless")
natestclassifier$properties = c("numerics", "factors", "oneclass", "twoclass", "multiclass", "prob", "weights", "missings")
natestclassifier$id = "classif.natest"
natestclassifier$packages = "llama"

othertestclassifier = lrn("classif.featureless")
othertestclassifier$properties = c("numerics", "factors", "oneclass", "twoclass", "multiclass", "prob")
othertestclassifier$id = "classif.natest"
othertestclassifier$packages = "llama"

foo = lrn("classif.featureless")
foo$properties = c("numerics", "factors", "twoclass", "multiclass", "prob")
foo$id = "classif.ftest"
foo$packages = "llama"

idtestclassifier = lrn("classif.featureless")
idtestclassifier$properties = c("numerics", "factors", "twoclass", "multiclass", "weights", "oneclass")
idtestclassifier$id = "classif.idtest"
idtestclassifier$packages = "llama"

probtestclassifier = lrn("classif.featureless")
probtestclassifier$properties = c("numerics", "factors", "twoclass", "multiclass", "weights", "prob")
probtestclassifier$id = "classif.ptest"
probtestclassifier$packages = "llama"
probtestclassifier$predict_type = "prob"

testregressor = lrn("regr.featureless")
testregressor$properties = c("numerics", "factors", "weights")
testregressor$id = "regr.test"
testregressor$packages = "llama"

algotestregressor <- lrn("regr.featureless")
algotestregressor$properties = c("numerics", "factors", "weights")
algotestregressor$id = "regr.test"
algotestregressor$packages = "llama"

threealgotestregressor <- lrn("regr.featureless")
threealgotestregressor$properties = c("numerics", "factors", "weights")
threealgotestregressor$id = "regr.test.algo.three"
threealgotestregressor$packages = "llama"

natestregressor <- lrn("regr.featureless")
natestregressor$properties = c("numerics", "factors", "weights", "missings")
natestregressor$id = "regr.natest"
natestregressor$packages = "llama"


footestregressor <- lrn("regr.featureless")
footestregressor$properties = c("numerics", "factors")
footestregressor$id = "regr.footest"
footestregressor$packages = "llama"

testclusterer <- lrn("clust.featureless")
testclusterer$properties = c("numerics")
testclusterer$id = "cluster.test"
testclusterer$packages = "llama"
