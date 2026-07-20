# Data Layout

This public repository keeps only the AEG-Edit mainline data assets.

## Retained

- `RustEvo/RustEvo.json`: core Rust API-evolution dataset
- `RustEvo/APIDocs.json`: RustEvo documentation file used by RAG-style baselines
- `PyEvo/PyEvo.json`: core Python API-evolution dataset
- `PyEvo/APIDocs.json`: PyEvo documentation file used by RAG-style baselines
- `EditEvery/editevery.json`: EditEvery subset from AnyEdit, used for non-executable auxiliary evaluation rather than execution-based benchmark results
- `rustevo_graphs/`: prebuilt RustEvo API Evolution Graphs
- `pyevo_graphs/`: prebuilt PyEvo API Evolution Graphs
- `alpaca_data.json`: auxiliary locality data used by some editing baselines

## Notes

- `EditEvery` originates from the [AnyEdit](https://github.com/jianghoucheng/AnyEdit) project.
- Unlike `RustEvo` and `PyEvo`, `EditEvery` is not used as an executable benchmark in this repository. It is only used for supplementary non-executable metrics.
