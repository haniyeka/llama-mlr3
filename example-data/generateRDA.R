library(llama)

features = read.csv("features.csv")

satsolvers = input(features,
          read.csv("times.csv"),
          read.csv("successes.csv"),
          costs=list(groups=list(all=tail(names(features), -1)), values=read.csv("feature-times.csv")))

save(satsolvers, file='satsolvers.rda', compress='xz')
