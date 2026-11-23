# Build and check environment for the llama R package (in code/).
#
# Build (from the repository root); R_VERSION is any rocker/r-ver tag:
#   docker build -t llama-check:4.6.2 .
#   docker build --build-arg R_VERSION=4.5.3 -t llama-check:4.5.3 .
#   docker build --build-arg R_VERSION=devel -t llama-check:devel .
#
# R packages come from the CRAN repository the rocker image is set up with:
#   - newest R release (4.6.2 as of 2026-11): current CRAN, Linux binaries;
#   - older releases (e.g. 4.5.3): a dated CRAN snapshot (binaries) taken
#     just before the next R release, so they always get the same versions;
#   - devel: current CRAN as source packages (slow: everything is compiled).
ARG R_VERSION=4.6.2
FROM rocker/r-ver:${R_VERSION}

# Java (headless JDK) for rJava and RWeka; headers for packages built from
# source; qpdf, pandoc and tidy are used by R CMD check --as-cran.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jdk-headless \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        qpdf \
        pandoc \
        tidy \
    && rm -rf /var/lib/apt/lists/* \
    && R CMD javareconf

WORKDIR /llama

# Dependencies in their own layer: only rebuilt when install_deps.R or
# code/DESCRIPTION change, not on every source change.
COPY install_deps.R ./
COPY code/DESCRIPTION code/DESCRIPTION
RUN Rscript install_deps.R

# The package source.
COPY code/ code/
