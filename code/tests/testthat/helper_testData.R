fold = data.frame(a=rep.int(0, 10), b=c(rep.int(1, 5), rep.int(0, 5)),
                  c=c(rep.int(0, 5), rep.int(1, 5)))
d = list(data=rbind(fold, fold),
         train=list(1:nrow(fold)),
         test=list(1:nrow(fold) + nrow(fold)),
         features=c("a"),
         minimize=T,
         performance=c("b", "c"),
         best=rep.int(c(rep.int("c", 5), rep.int("b", 5)), 2))
class(d) = "llama.data"
attr(d, "hasSplits") = TRUE


folde = data.frame(a=c(rep.int(0, 5), rep.int(1, 5)),
                  b=c(rep.int(1, 5), rep.int(0, 5)),
                  c=rep.int(1, 10))
e = list(data=rbind(folde, folde),
         train=list(1:nrow(folde)),
         test=list(1:nrow(folde) + nrow(folde)),
         features=c("c"), minimize=T,
         performance=c("a", "b"),
         best=rep.int("b", 20))
class(e) = "llama.data"
attr(e, "hasSplits") = TRUE


foldf = data.frame(a=rep.int(0, 10), b=c(rep.int(1, 5), rep.int(0, 5)),
                  c=c(rep.int(0, 5), rep.int(1, 5)))
dnosplit = list(data=rbind(foldf, foldf),
         features=c("a"),
         minimize=T,
         performance=c("b", "c"),
         best=rep.int(c(rep.int("c", 5), rep.int("b", 5)), 2))
class(dnosplit) = "llama.data"


foldg = data.frame(a=rep.int(0, 10), b=rep.int(1, 10), c=rep.int(0, 10),
                   d=rep.int(T, 10), e=rep.int(F, 10))
g = list(data=rbind(foldg, foldg),
         train=list(1:nrow(foldg)),
         test=list(1:nrow(foldg) + nrow(foldg)),
         features=c("a"),
         performance=c("b", "c"),
         success=c("d", "e"),
         minimize=T,
         best=rep.int("b", 20))
class(g) = "llama.data"
attr(g, "hasSplits") = TRUE

bests = c("a", "a", "a", "b", "b")
bestlist = list("a", "b", c("a", "b"))
bestlistlong = list("a", "a", "b", c("a", "b"), "a", "a", c("a", "b"), "a", "a", "a")
