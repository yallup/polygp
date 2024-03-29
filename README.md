# polygp
A wrapper around polychord for fitting spectral kernels with nested sampling

### Installation
Clone the repository and install with

```
pip install .
```

requires `pypolychord` interface (not on pypi) to be built and installed, see instructions at [github.com/PolyChord/PolyChordLite](https://github.com/PolyChord/PolyChordLite)

### Tests
Uses `pytest` as the test framework, tests defined in the `tests` directory, to run the suite run the following in the top of the repository
```
python -m pytest
```

Pre-commit hooks to enforce consistent code style are used in this repo, if installed run 
```
pre-commit
```
At the top level of the repo to configure this.

### Usage

An example notebook is included in `tutorial.ipynb`

More involved examples are in the `examples` directory
