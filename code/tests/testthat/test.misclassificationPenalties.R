test_that("misclassificationPenalties returns penalties", {
    fold = data.frame(a=rep.int(1, 5), b=rep.int(0, 5))
    d = list(test=list(fold, fold), performance=c("a", "b"), minimize=T)
    as = rep.int(list(data.frame(algorithm="a", score=1)), 5)
    bs = rep.int(list(data.frame(algorithm="b", score=1)), 5)
    modela = list(predictions=list(as, as))
    class(modela) = "llama.model"
    attr(modela, "hasPredictions") = TRUE
    modelb = list(predictions=list(bs, bs))
    class(modelb) = "llama.model"
    attr(modelb, "hasPredictions") = TRUE

    expect_equal(sum(misclassificationPenalties(d, modela)), 10)
    expect_equal(sum(misclassificationPenalties(d, modelb)), 0)
})

test_that("misclassificationPenalties works without test split", {
    fold = data.frame(a=rep.int(1, 10), b=rep.int(0, 10))
    d = list(data=fold, performance=c("a", "b"), minimize=T)
    modela = list(predictions=rep.int(list(data.frame(algorithm="a", score=1)), 10))
    class(modela) = "llama.model"
    attr(modela, "hasPredictions") = TRUE
    modelb = list(predictions=rep.int(list(data.frame(algorithm="b", score=1)), 10))
    class(modelb) = "llama.model"
    attr(modelb, "hasPredictions") = TRUE

    expect_equal(sum(misclassificationPenalties(d, modela)), 10)
    expect_equal(sum(misclassificationPenalties(d, modelb)), 0)
})

test_that("misclassificationPenalties allows to maximise", {
    fold = data.frame(a=rep.int(1, 5), b=rep.int(0, 5))
    d = list(test=list(fold, fold), performance=c("a", "b"), minimize=F)
    as = rep.int(list(data.frame(algorithm="a", score=1)), 5)
    bs = rep.int(list(data.frame(algorithm="b", score=1)), 5)
    modela = list(predictions=list(as, as))
    class(modela) = "llama.model"
    attr(modela, "hasPredictions") = TRUE
    modelb = list(predictions=list(bs, bs))
    class(modelb) = "llama.model"
    attr(modelb, "hasPredictions") = TRUE

    expect_equal(sum(misclassificationPenalties(d, modela)), 0)
    expect_equal(sum(misclassificationPenalties(d, modelb)), 10)
})

test_that("misclassificationPenalties works for test splits", {
    fold = data.frame(a=rep.int(1, 5), b=rep.int(0, 5),
        d=rep.int(F, 5), e=rep.int(T, 5))
    d = list(data=rbind(fold, fold), test=list(fold), performance=c("a", "b"), success=c("d", "e"), minimize=T)
    model = function(data) {
        lapply(1:nrow(data$data), function(i) { data.frame(algorithm="a", score=1) })
    }
    class(model) = "llama.model"
    attr(model, "hasPredictions") = FALSE

    expect_equal(sum(misclassificationPenalties(d, model)), 5)
})

test_that("misclassificationPenalties works with NA predictions", {
    fold = data.frame(a=rep.int(1, 5), b=rep.int(0, 5),
        d=rep.int(F, 5), e=rep.int(T, 5))
    d = list(data=rbind(fold, fold), test=list(fold, fold), performance=c("a", "b"), success=c("d", "e"), minimize=T)
    nas = rep.int(list(data.frame(algorithm=NA, score=0)), 5)
    as = rep.int(list(data.frame(algorithm="a", score=1)), 5)
    model = list(predictions=list(nas, as))
    class(model) = "llama.model"
    attr(model, "hasPredictions") = TRUE

    mcps = misclassificationPenalties(d, model)
    expect_equal(length(mcps), 10)
    expect_equal(sum(mcps, na.rm=T), 5)
})
