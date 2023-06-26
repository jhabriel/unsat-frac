[![DOI](https://zenodo.org/badge/333197352.svg)](https://zenodo.org/badge/latestdoi/333197352)

# unsat-frac: A PorePy extension package for simulating the saturated-unsaturated flow in fractured porous media

**unsat-frac** is a Python package created as an extension of [PorePy](https://github.com/pmgbergen/porepy) for modeling and simulation of the unsaturated 
flow with fractures acting as capillary barriers.

## Citing

If you use **unsat-frac** in your research, we ask you to cite the following reference:

*Add pre-print reference*

## Installation from source

**unsat-frac** is developed under Python >= 3.9. Get the latest version by cloning this repository, i.e.:

    git clone https://github.com/jhabriel/unsat-frac.git
    cd unsat-frac

Now, install the dependencies:

     pip install -r requirements.txt

**unsat-fract** requires PorePy (commit b5ddf54cf5e71ee96ec8526234f24a1ee6a81a1c)
to be installed. If you do not have PorePy installed, please 
[do so](https://github.com/pmgbergen/porepy/blob/develop/Install.md) before installing **unsat-frac**.

To install **unsat-frac**:

    pip install .

Or, for user-editable installations:

    pip install --editable .

## Examples

All the numerical examples included in the manuscript can be found in the **paper_examples** folder.

## Troubleshooting
Create an [issue](https://github.com/jhabriel/unsat-frac).

## License
See [license md](./LICENSE.md).
