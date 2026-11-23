# Install everything needed to build, check and test the llama (mlr3) package.
#
# Usage (from the repository root):
#   Rscript install_deps.R [path/to/DESCRIPTION]     # default: code/DESCRIPTION
#
# The package list is read from DESCRIPTION (Depends, Imports, LinkingTo,
# Suggests), plus the learner backends the tests and examples need (below).
#
# Packages come from the repositories R is configured with
# (getOption("repos")). In the rocker Docker images that is a dated binary
# snapshot, so image builds are fast and reproducible. If no CRAN mirror is
# configured, https://cloud.r-project.org is used. mlr3extralearners is not on
# CRAN; it comes from the mlr-org r-universe, after everything else, so the
# other packages always come from CRAN.
#
# Packages that are already installed are not reinstalled or upgraded.
#
# System requirements: a JDK (for rJava/RWeka; run `sudo R CMD javareconf`
# if rJava fails to find Java), C/C++/Fortran compilers, and the
# libcurl/libssl/libxml2 development headers.

desc_file <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(desc_file)) desc_file <- "code/DESCRIPTION"
if (!file.exists(desc_file)) stop("DESCRIPTION not found: ", desc_file)

# Learner backends that are not in DESCRIPTION.
extra_pkgs <- c(
  # tests: classif.J48, classif.OneR, clust.SimpleKMeans (RWeka);
  #        classif.ranger (ranger)
  "RWeka", "ranger",
  # examples in man/: classif.randomForest, classif.svm (e1071),
  #                   regr.ksvm (kernlab)
  "randomForest", "e1071", "kernlab",
  # tuning with mlr3 (replaces ParamHelpers designs, PLAN step 5)
  "mlr3tuning", "paradox"
)
# Not on CRAN.
universe_pkgs <- c(mlr3extralearners = "https://mlr-org.r-universe.dev")

# Install into the first library R uses if it is writable (in Docker: the
# site library; on a desktop: usually the per-user library), otherwise into
# the per-user library.
lib <- .libPaths()[1]
if (file.access(lib, 2) != 0) {
  lib <- path.expand(strsplit(Sys.getenv("R_LIBS_USER"), .Platform$path.sep)[[1]][1])
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
cores <- parallel::detectCores()
options(Ncpus = if (is.na(cores)) 1L else max(1L, cores - 1L))

repos <- getOption("repos")
if (is.null(repos) || is.na(repos["CRAN"]) || repos["CRAN"] == "@CRAN@") {
  repos["CRAN"] <- "https://cloud.r-project.org"
}

# Package names from DESCRIPTION, without version requirements.
fields <- read.dcf(desc_file, fields = c("Depends", "Imports", "LinkingTo", "Suggests"))
desc_pkgs <- trimws(sub("\\(.*", "", unlist(strsplit(fields[!is.na(fields)], ","))))
base_pkgs <- rownames(installed.packages(priority = "base"))
desc_pkgs <- setdiff(desc_pkgs[nzchar(desc_pkgs)], c("R", base_pkgs))

all_pkgs <- unique(c(desc_pkgs, extra_pkgs, names(universe_pkgs)))
missing_pkgs <- function() setdiff(all_pkgs, rownames(installed.packages()))

message("R ", getRversion(), "; installing into ", lib)
message("Repositories: ", paste(repos, collapse = ", "))

# 1. Everything that is on CRAN.
todo <- setdiff(missing_pkgs(), names(universe_pkgs))
if (length(todo)) install.packages(todo, lib = lib, repos = repos)

# 2. Packages from r-universe (their CRAN dependencies are already in place).
todo <- intersect(missing_pkgs(), names(universe_pkgs))
if (length(todo)) {
  install.packages(todo, lib = lib, repos = c(unique(universe_pkgs[todo]), repos))
}

still_missing <- missing_pkgs()
if (length(still_missing)) {
  stop("Failed to install: ", paste(still_missing, collapse = ", "))
}
message("All ", length(all_pkgs), " llama dependencies are installed.")
