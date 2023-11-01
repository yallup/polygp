# polygp
A wrapper around polychord for fitting spectral kernels with nested sampling

# Installation
Clone the repository and install with 

```
pip install .
```

requires `pypolychord` interface (not on pypi) to be built and installed, see instructions at [github.com/PolyChord/PolyChordLite](https://github.com/PolyChord/PolyChordLite)

# Tests
Uses `pytest` as the test framework, tests defined in the `tests` directory, to run the suite run the following in the top of the repository
```
python -m pytest
```

# Usage

A simple sythetic data example is included at `examples/test_sin.py`, it is recommended to run this using mpi.
This runs a fit of a Spectral Mixture on a dataset with 4 identifiable true frequency modes, the output diagnostic plots are saved in the usual `chains` folder

A set of benchmark real data examples can be run through `examples/benchmark_datasets.py` however some of these take a long time due to the number of data points
