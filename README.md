# DWCal

DWCal, or Delay-Weighted Calibration, is a new technique for precision interferometric calibration.
To learn more about this approach, read [Byrne 2022](https://arxiv.org/abs/2208.04406).

## Installation

Run `pip install -e .` from the top-level directory.

## Dependencies

- scipy
- [pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata)

## Getting started

Check out `tutorial.ipynb` for an example of calibrating with and without delay weighting.
Sample simulated data is available as uvfits files in the `dwcal/data/` directory.
`data.uvfits` contains visibilities simulated from a full sky catalog, and `model.uvfits`
contains visibilities simulated from an incomplete sky model. See [Byrne 2022](https://arxiv.org/abs/2208.04406)
for more details about these simulations and the calibration results.
