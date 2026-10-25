# IndPenSim Pilot Dataset Report

## Provenance

- Simulator source: `https://github.com/Lemniscabio/IndPenSim`, local commit `45c3d8c53ba423671acfb63de5855ecd4c8aa9d4`.
- Measurement source: public CC BY 4.0 IndPenSim compilation at `foundation-models/golden-batch-sentinel-data`, derived from Mendeley DOI `10.17632/pdnjz7zz5x.2`.
- `batches.parquet`: 6,705,383 bytes; SHA-256 `704843412cdf370a54fb19c72f5bbeeeb7989195607930dee9a2eb062acfa124`.
- `statistics.parquet`: 5,860 bytes; SHA-256 `e6df6081c6e500784a808fdf992334f020fb5355a0a3d6ac30a872be400ffe4c`.

These are genuine simulator-generated measurements, not synthetic examples created by MaverickX.

## Measured contents

- 113,935 time-series rows and 37 source columns.
- 100 batches identified by 100 elapsed-time resets and cross-checked against the 100 ordered statistics records.
- 90 fault-free batches and 10 documented fault batches.
- Nominal sampling interval: 0.2 h (12 minutes).
- Batch lengths: 835 to 1,450 rows (167 to 290 h).
- Required online measurements (dissolved oxygen, pH, temperature, pressure) have no missing values.
- Five offline laboratory channels contain 111,873 missing values each because they are sampled only about every 12 h; they are preserved but not used by the first model.

Fault batches contain four categories:

| Source batches | Category | Known active intervals (h) |
|---|---|---|
| 91, 95, 97 | Substrate feed | 20–30, 76–92, 200–214 (truncated when a batch ends early) |
| 92, 98 | Base flow | 80–84, 140–160 |
| 93, 96 | Coolant flow | 70–90 |
| 94, 99, 100 | Aeration | 20–24, 100–110 |

## Source-quality issue and documented handling

The parquet compilation contains columns named `Batch ID` and `Fault flag` whose values are not usable identifiers or binary flags; they vary like large process/spectral measurements. The source's `Batch reference` is only a normal/fault campaign indicator. The loader therefore reconstructs stable source batch numbers 1–100 from ordered time resets, then verifies the count and order against `statistics.parquet`. It uses the per-row `Fault reference` channel for exact active intervals and the statistics table only for the batch-level normal/fault label.

No fault label, batch identifier, control-strategy field, or fault-active flag is used as a model feature.

