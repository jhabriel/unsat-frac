[![DOI](https://zenodo.org/badge/253091118.svg)](https://zenodo.org/badge/latestdoi/253091118)


# unsat-frac: Unsaturated flow in fractured porous media

**unsat-frac** is a Python package created as an extension of [PorePy](https://github.com/pmgbergen/porepy) for modeling and simulation of the unsaturated flow in the presence of air-filled fractures.

## Citing

If you use **unsat-frac** in your research, we ask you to cite the following reference:

*Add reference*

## Installation from source

**unsat-frac** is developed under Python >= 3.7. Get the latest version by cloning this repository, i.e.:

    git clone https://github.com/jhabriel/unsat-frac.git
    cd unsat-frac

Now, install the dependencies:

     pip install -r requirements.txt

We require the development version of PorePy >= X.X.X to be installed. If you do not have PorePy installed, please [do so](https://github.com/pmgbergen/porepy/blob/develop/Install.md) before installing **unsat-frac**.

To install **unsat-frac**:

    pip install .

Or, for user-editable installations:

    pip install --editable .

## Getting started

A simple usage of **unsat-frac** can be found in tutorials/xxx.ipynb.

## Examples

All the numerical examples included in the manuscript can be found in the **paper_examples** folder. These include two validation cases and two benchmark problems.

## Problems, suggestions, enhancements...
Create an [issue](https://github.com/jhabriel/unsat-frac).

## License
See [license md](./LICENSE.md).
