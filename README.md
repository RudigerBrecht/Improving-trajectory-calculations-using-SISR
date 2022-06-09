# Improving-trajectory-calculations-using-SISR

Lagrangian trajectory or particle dispersion models require meteorological data such as wind, tem-
perature and geopotential at the exact spatio-temporal locations of the particles that move in-
dependently from a regular grid. Traditionally, this high-resolution data has been obtained by
interpolating the meteorological parameters from the gridded output of a meteorological model or
re-analysis, e.g. using linear interpolation in space and time. However, interpolation errors are a
large source of error for these models. Reducing them requires meteorological input fields with high
space and time resolution, which may not always be available and causes severe data storage and
transfer problems. Here, we interpret this problem as a single image superresolution task. That
is, we interpret meteorological fields available at their native resolution as low-resolution images
and train deep neural networks to up-scale them to higher resolution, thereby providing more accu-
rate data for particle dispersion models. We train various versions of the state-of-the-art Enhanced
Deep Residual Networks for Superresolution (EDSR) on low-resolution ERA5 data with the goal to
up-scale these data to arbitrary resolution. We show that the resulting up-scaled wind fields have
root-mean-squared errors 50% smaller than winds obtained with linear interpolation at compara-
ble computational inference costs. In a test set-up using the Lagrangian particle dispersion model
FLEXPART and reduced-resolution wind fields, we demonstrate that absolute horizontal transport
deviations of calculated trajectories from "ground-truth" trajectories calculated with undegraded
high-resolution winds are reduced by XXX% relative to trajectories using linear interpolation of the
wind data.


# About this repository 

Once the manuscript ([link](https://arxiv.org/abs/2206.04015)) is accapted, we will upload the code to train the neural networks.
For now, the code in this repository reproduces the plots from the manuscript. To run the jupyter notebook you need to download the data from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6628749.svg)](https://doi.org/10.5281/zenodo.6628749).
