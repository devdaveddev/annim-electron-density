# ANNIM: global ionospheric electron density with one neural network

One feed-forward PyTorch network that predicts electron density (Ne) at any place, time, altitude and space-weather condition, from 100 to 850 km. It is trained on ionosonde and satellite radio-occultation data and tested on whole years the network never saw. The notebook also reproduces the figures of Gowtam et al. (2019, JGR Space Physics, ANNIM-3D) for comparison.

## What is in this repo

- `ne_v6b.ipynb`: the full pipeline (v6b). Part A converts the data, splits it by whole years, trains the network and scores it. Part B draws the paper figures, then an appendix adds extra diagnostics and physics checks.

## Method

- **Data:** GIRO Digisonde profiles (bottomside), COSMIC-1 and COSMIC-2 radio occultation (topside). v6b uses GIRO and COSMIC-1 in full and 30% of COSMIC-2 profiles.
- **Split:** whole years are kept together (train, validation, test, plus a future year), so test rows never sit next to training rows in time.
- **Inputs (19):** day of year, UT, local time and longitude as sin/cos, latitude, altitude, Sym-H, Kp, F10.7 and its 81-day mean, polar-cap potential, and mean Sym-H over the previous 3, 6, 12 and 24 hours. Target: log10 Ne.
- **Network:** smooth-activation MLP trained by streaming 2-million-row blocks from disk, with each source and year weighted so the large COSMIC-2 years do not dominate.
- **Training changes in v6b:** a 3-epoch schedule with the smoothness penalty ramped in, a penalty so NmF2 never falls when F10.7 rises, and extra weight on rare high-flux rows.
- **Checks:** data sanity checks, physics sanity checks (equatorial anomaly crests, NmF2 against F10.7, storm response), leave-one-year-out runs, and a comparison against v5.

## Results (v6b, unseen test years)

| Measure | v6b |
| --- | --- |
| R2 on log Ne | 0.813 |
| R2 on linear Ne | 0.832 |
| Typical miss (factor) | x1.95 |
| Bias (factor) | x1.09 |
| Test RMSE (log10 Ne) | 0.289 |

v5 (21 inputs, with dip latitude and drift) scored R2 0.780 on log Ne and a typical miss of x2.06 on the same rows. Known weak points: strong storms and the bottomside at night, where the two data sources disagree.

## Stack

Python, PyTorch, NumPy, SciPy, PyArrow (parquet), h5py, Matplotlib, Cartopy, PyIRI (IRI baseline).

## How to run

1. Install: `pip install torch numpy scipy pyarrow h5py matplotlib cartopy PyIRI`.
2. Get the data (not included, it is about 150 GB): monthly GIRO, COSMIC-1 and COSMIC-2 `.mat` files, plus Kp, Sym-H and F10.7 series.
3. Edit the paths in section 0 of the notebook, run section 0b once to convert the `.mat` files to parquet (it can resume), then run the cells in order.
4. Part A trains the network and prints the metrics. Part B draws the figures.

## Notes

- Data and trained weights are not in this repo.
- The notebook stores its outputs, so you can read the results without re-running it.
- Author: Devansh Misra, summer research at the Indian Institute of Geomagnetism.
