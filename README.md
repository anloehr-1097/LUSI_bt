
# Learning Using Statistical Invariants

This repository contains an implementation of the LUSI paradigm for binary classification on 2 MNIST classes. (7's and 8's).
It heavily depends on the tensorflow library.

The repo accomodates the following files:

- lusi_Andreas_Loehr.py
- lusi_periphery.py
- lusi_demo.ipynb
- lusi_env.yml


The _lusi_Andreas_Loehr.py_ file:
This file is the centerpiece of the implementation.
In here, an implementation of two classes for training models in the LUSI framework as well as in the ERM-LUSI framework is provided.

The _lusi_periphery.py_ file:
This file hosts an important class to handle data and process it such that it can be used with the 2 main classes from _lusi_Andreas_Loehr.py_.
Moreover, additional functions for numerous tasks which emerged during the creation of the implementation are included.

The _lusi_demo.ipynb_ file:
This notebook demonstrates how to use the classes from the .py files. A model is trained in each of the frameworks and evaluated on a test set.
Furthermore, it features some illustrations in the form of plots to improve the understanding of the concept of a predicate.

The _lusi_env.yml_ file:
This file is the result of an export of the env used to write up the implementation. It can be used to create an anaconda env with the required packages.



