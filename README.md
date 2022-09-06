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


# Preprint

For the manuscript see ([link](https://arxiv.org/abs/2206.04015)).

# Reproducing the results

1. Download the ERA5 re-analysis data from the Copernicus Climate
Change Service (C3S) Climate Data Store. The results contain modified Copernicus Climate Change Service information. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information
or data it contains.
We used flex_extract to download the ERA5 re-analysis data. The documentation can be found here
(https://www.flexpart.eu/flex_extract/ecmwf_data.html).

2. Normalize the downloaded data using the scripts in the folder *preprocess_data*.

3. Train the neural networks using the scripts in the folder *train_neural_networks*.

4. To interpolate the data first copy the evaluation data as follows:
```
cp /download_era5_data/eval_u.nc /interpolate_data/nn_eval_u_m1.nc
cp /download_era5_data/eval_v.nc /interpolate_data/nn_eval_v_m1.nc

cp /download_era5_data/eval_u.nc /interpolate_data/nn_eval_u_m2.nc
cp /download_era5_data/eval_v.nc /interpolate_data/nn_eval_v_m2.nc

cp /download_era5_data/eval_u.nc /interpolate_data/nn_eval_u_m4.nc
cp /download_era5_data/eval_v.nc /interpolate_data/nn_eval_v_m4.nc

cp /download_era5_data/eval_u.nc /interpolate_data/lin_eval_u.nc
cp /download_era5_data/eval_v.nc /interpolate_data/lin_eval_v.nc
```
Then interpolate the data using the scripts in the folder *interpolate_data*.

5. Evaluate the interpolated data using the scripts in the folder *evaluate_interpolation*

6. Then we used the Lagrangian transport and dispersion model FLEXPART ([link](https://www.flexpart.eu/)) to advect particles on the interpolated data.

7. Plot the resluts using the jupyter notebook with the data from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6628749.svg)](https://doi.org/10.5281/zenodo.6628749).
