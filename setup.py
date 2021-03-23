"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="mdunsat",  # Required
    version="0.0.1",  # Required
    description="Unsaturated flow in fractured porous media",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/jhabriel/unsat-frac",  # Optional
    author="Jhabriel Varela",  # Optional
    author_email="jhabriel.varela@uib.no",  # Optional
    keywords="unsat-frac, porepy, fracture, porous media",  # Optional
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.6, <4",
    install_requires=required,  # Optional
    extras_require={"dev": ["check-manifest"], "test": ["coverage"]},  # Optional
    package_data={"mdunsat": ["py.typed"]},  # Optional
)
