## [Passive membranes](Passive membrane.ipynb)

This notebook illustrates how the membrane can be modeled by an RC circuit with a battery. It shows the results of the analytical solution and also demonstrates how the differential equation can be integrated using Euler integration. I also discuss my struggles on how to get the units right for the equations.

## [Hodgkin Huxley model](Hodgkin Huxley.ipynb)
Probably the most famous model in neuroscience! In this notebook first show how ion channels are modelled using alpha and beta functions. Then I implement the full Hodgin-Huxley model

## [Simplified neural models](Simplified models.ipynb)
Here I provide code for the integrate and fire model and the exponential integrate and fire model. Both models approximate the behavior of neurons with a single differential equation. While the integrate and fire model is basically an RC circuit with some additional thresholds, the exponential integrate and fire model is already able to simulate some properties of action potential generation.