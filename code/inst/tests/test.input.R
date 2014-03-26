test_that("input reads features and performances", {
    a = data.frame(a=c(1:5), b=rep.int(1, 5))
    b = data.frame(a=c(1:5), c=rep.int(1, 5))

    data = input(a, b)
    expect_equal(data$features, c("b"))
    expect_equal(data$performance, c("c"))
    expect_equal(data$success, c())
    expect_equal(dim(data$data), c(5, 4))
    expect_equal(data$data$best, factor(rep.int("c", 5)))
})

test_that("input determines best", {
    a = data.frame(a=c(1:5), b=rep.int(1, 5))
    b = data.frame(a=c(1:5), c=rep.int(1, 5), d=rep.int(2, 5))

    data = input(a, b)
    expect_equal(data$features, c("b"))
    expect_equal(data$performance, c("c", "d"))
    expect_equal(data$data$best, factor(rep.int("c", 5)))
})

test_that("input determines best with max", {
    a = data.frame(a=c(1:5), b=rep.int(1, 5))
    b = data.frame(a=c(1:5), c=rep.int(1, 5), d=rep.int(2, 5))

    data = input(a, b, minimize=F)
    expect_equal(data$features, c("b"))
    expect_equal(data$performance, c("c", "d"))
    expect_equal(data$data$best, factor(rep.int("d", 5)))
})

test_that("input reads features, performances and successes", {
    a = data.frame(a=c(1:5), b=rep.int(1, 5))
    b = data.frame(a=c(1:5), c=rep.int(1, 5))

    data = input(a, b, b)
    expect_equal(data$features, c("b"))
    expect_equal(data$performance, c("c"))
    expect_equal(data$success, c("c_success"))
    expect_equal(dim(data$data), c(5, 5))
    expect_equal(data$data$best, factor(rep.int("c", 5)))
})

test_that("input allows to specify single cost for all", {
    a = data.frame(a=c(1:5), b=rep.int(1, 5))
    b = data.frame(a=c(1:5), c=rep.int(1, 5))

    data = input(a, b, costs=3)
    expect_equal(data$features, c("b"))
    expect_equal(data$performance, c("c"))
    expect_equal(data$cost, c("cost"))
    expect_equal(data$data$cost, rep.int(3, 5))
})

test_that("input allows to costs", {
    a = data.frame(a=c(1:5), b=rep.int(1, 5))
    b = data.frame(a=c(1:5), c=rep.int(1, 5))

    data = input(a, b, costs=data.frame(a=c(1:5), e=c(1:5)))
    expect_equal(data$features, c("b"))
    expect_equal(data$performance, c("c"))
    expect_equal(data$cost, c("e"))
    expect_equal(data$data$e, c(1:5))
})

test_that("input warns about different number of rows", {
    a = data.frame(a=c(1:6), b=rep.int(1, 6))
    b = data.frame(a=c(1:5), c=rep.int(1, 5))

    expect_warning(input(a, b), "Different number of rows in data frames, taking only common rows.")
})

test_that("input errors when not being able to link", {
    a = data.frame(a=rep.int(1, 5))
    b = data.frame(b=rep.int(1, 5))

    expect_error(input(a, b), "Performance can't be linked to features -- no common columns!")
})

test_that("input errors with non-unique IDs", {
    a = data.frame(a=rep.int(1, 5), b=rep.int(1, 5))
    b = data.frame(a=rep.int(1, 5), c=rep.int(1, 5))

    expect_error(input(a, b), "Common columns do not provide unique IDs!")
})
