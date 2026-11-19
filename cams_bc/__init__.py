"""CAMS-to-measurement correction for unseen locations (spatial transfer).

Sister package of `solar_bc` (imputation at the measuring stations). Same data (DWD 10-min GHI/DHI
at 94 stations, CAMS Radiation Service at the station point and on the 3/8 km ring, ERA5 solar and
met caches), different question: can a pointwise model trained at measuring stations sharpen CAMS at
a location WITHOUT measurements (a PV park)? Hence a strict station split: test stations are never
seen in training, and the 3-fold cross-validation is over disjoint station groups, not over
(station, day) cells as in solar_bc.
"""
