test_that("parscores returns parscores", {
    fold = data.frame(a=rep.int(1, 5), b=rep.int(0, 5),
        d=rep.int(F, 5), e=rep.int(T, 5))
    d = list(test=list(fold, fold), performance=c("a", "b"), success=c("d", "e"))
    as = rep.int(list(data.frame(algorithm="a", score=1)), 5)
    bs = rep.int(list(data.frame(algorithm="b", score=1)), 5)
    modela = list(predictions=list(as, as))
    modelb = list(predictions=list(bs, bs))

    expect_equal(sum(parscores(d, modela)), 100)
    expect_equal(sum(parscores(d, modelb)), 0)
})

test_that("parscores raises error if no successes are given", {
    fold = data.frame(a=rep.int(1, 5), b=rep.int(0, 5))
    d = list(test=list(fold, fold), performance=c("a", "b"))
    as = rep.int(list(data.frame(algorithm="a", score=1)), 5)
    modela = list(predictions=list(as, as))
    expect_error(parscores(d, modela), "Need successes to calculate PAR scores!")
})

test_that("parscores works without test split", {
    fold = data.frame(a=rep.int(1, 10), b=rep.int(0, 10),
        d=rep.int(F, 10), e=rep.int(T, 10))
    d = list(data=fold, performance=c("a", "b"), success=c("d", "e"))
    modela = list(predictions=rep.int(list(data.frame(algorithm="a", score=1)), 10))
    modelb = list(predictions=rep.int(list(data.frame(algorithm="b", score=1)), 10))

    expect_equal(sum(parscores(d, modela)), 100)
    expect_equal(sum(parscores(d, modelb)), 0)
})

test_that("parscores allows to specify factor", {
    fold = data.frame(a=rep.int(1, 5), b=rep.int(0, 5),
        d=rep.int(F, 5), e=rep.int(T, 5))
    d = list(test=list(fold, fold), performance=c("a", "b"), success=c("d", "e"))
    as = rep.int(list(data.frame(algorithm="a", score=1)), 5)
    bs = rep.int(list(data.frame(algorithm="b", score=1)), 5)
    modela = list(predictions=list(as, as))
    modelb = list(predictions=list(bs, bs))

    expect_equal(sum(parscores(d, modela, 100)), 1000)
    expect_equal(sum(parscores(d, modelb, 100)), 0)
})

test_that("parscores allows to specify timeout", {
    fold = data.frame(a=rep.int(1, 5), b=rep.int(0, 5),
        d=rep.int(F, 5), e=rep.int(T, 5))
    d = list(test=list(fold, fold), performance=c("a", "b"), success=c("d", "e"))
    as = rep.int(list(data.frame(algorithm="a", score=1)), 5)
    bs = rep.int(list(data.frame(algorithm="b", score=1)), 5)
    modela = list(predictions=list(as, as))
    modelb = list(predictions=list(bs, bs))

    expect_equal(sum(parscores(d, modela, timeout=5)), 500)
    expect_equal(sum(parscores(d, modelb, timeout=5)), 0)
})
