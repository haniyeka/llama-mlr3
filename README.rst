=============================================================
LLAMA: Leveraging Learning to Automatically Manage Algorithms
=============================================================

LLAMA is an R package for algorithm portfolios and selection. It does not
provide any actual machine learning algorithms, but rather the infrastructure
required to use those in an algorithm selection context. There are functions to
create the most common types of algorithm selection models used in the
literature.

If you have any questions or feedback, please contact Lars Kotthoff
<larsko@uwyo.edu>.

Quick start
===========

So you know about algorithm portfolios and selection and just want to get
started. Here we go. In your R shell, type

::

    install.packages("llama")
    require(llama)

to install and load LLAMA. We're going to assume that you have two input CSV
files for your data -- features and times. The rows designate problem instances
and the columns feature and solver names. All files have an 'ID' column that
allows to link them. Load them into the data structure required by LLAMA as
follows.

::

    data = input(read.csv("features.csv"), read.csv("times.csv"))

You can also use the SAT solver data that comes with LLAMA by running

::

    data(satsolvers)
    data = satsolvers

Now partition the entire set of instances into training and test sets for
cross-validation.

::

    folds = cvFolds(data)

This will give you 10 folds for cross-validation. Now we're ready to train
our first model. To do that, we'll need some machine learning algorithms --
LLAMA is integrated with `mlr <https://github.com/berndbischl/mlr>`_ and
supports all its learning algorithms. We're going to use a random forest here
and train a simple classification model that predicts the best algorithm.

::

    model = classify(lrn("classif.randomForest"), folds)

Great! Now let's see how well this model is doing and compare its performance to
the virtual best solver (VBS) and the single best solver in terms of average
misclassification penalty.

::

    mean(misclassificationPenalties(data, vbs))
    mean(misclassificationPenalties(folds, model))
    mean(misclassificationPenalties(data, singleBest))

These are the numbers I get for the ``satsolvers`` data:

:virtual best: 0
:model: 74.73368
:single best: 122.3186

While we are quite far off the virtual best, our classifier beats the single
best algorithm! Not bad for a model trained in a single line of code.

You can use any other classification algorithms instead of a random forest of
course. You can also train regression or cluster models, use different
train/test splits or preprocess the data by selecting the most important
features. More details in the on-line documentation and the manual.

More information
================

More information can be found in the manual at http://arxiv.org/abs/1306.1031,
which is also included in the R package. In addition, there are R help pages for
all functions.

If you find LLAMA helpful, it would be great if you could cite the manual in any
publications!

::

    @techreport{kotthoff_llama_2013,
        address = {{arXiv}},
        title = {{LLAMA:} Leveraging Learning to Automatically Manage Algorithms},
        url = {http://arxiv.org/abs/1306.1031},
        number = {{arXiv:1306.1031}},
        author = {Kotthoff, Lars},
        month = jun,
        year = {2013}
    }
