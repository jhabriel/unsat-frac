# unsat-frac: A PorePy extension package for simulating the unsatured flow in fractured porous media
<img src="show.gif" width="700">

**Animation:** Fast flow of water through the fracture.

**unsat-frac** is a Python package created as an extension of [PorePy](https://github.com/pmgbergen/porepy) for modeling and simulation of the unsaturated flow in the presence of fractures acting as capillary barriers.

## Citing

If you use **unsat-frac** in your research, we ask you to cite the following reference:

*Add reference*

## Installation from source

**unsat-frac** is developed under Python >= 3.9. Get the latest version by cloning this repository, i.e.:

    git clone https://github.com/jhabriel/unsat-frac.git
    cd unsat-frac

Now, install the dependencies:

     pip install -r requirements.txt

**unsat-fract** requires the latest developer version of PorePy. If you do not have PorePy installed, please [do so](https://github.com/pmgbergen/porepy/blob/develop/Install.md) before installing **unsat-frac**.

To install **unsat-frac**:

    pip install .

Or, for user-editable installations:

    pip install --editable .

## Examples

All the numerical examples included in the manuscript can be found in the **paper_examples** folder. For the moment, tested numerical examples include 01_2d_vertical_fracture and 02_2d_junction

## Troubleshooting
Create an [issue](https://github.com/jhabriel/unsat-frac).

## License
See [license md](./LICENSE.md).
