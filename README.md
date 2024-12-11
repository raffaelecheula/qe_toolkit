# QE-Toolkit: Utilities for Quantum Espresso Calculations

## Overview

`qe_toolkit` is a Python package that simplifies the setup, execution, and analysis of Quantum Espresso simulations. It provides tools for managing input and output files, automating workflows, and extracting relevant data from simulation results.

## Features

- **Input File Generation**: Tools to create and validate Quantum Espresso input files with ease.
- **Job Management**: Scripts for submitting and monitoring simulation jobs on different computational infrastructures.
- **Data Extraction**: Utilities to parse and extract data from Quantum Espresso output files.
- **Post-Processing**: Functions to analyze results, including band structures, density of states, and charge densities.

## Installation

To install the package, clone the repository and use `pip`:

```bash
git clone https://github.com/raffaelecheula/qe_toolkit.git
cd qe_toolkit
pip install -e .
```

Requirements:
- Python 3.5 or later
- Numpy
- [ASE](https://wiki.fysik.dtu.dk/ase/)

## Usage

Examples of how to use `qe_toolkit` are in the `examples` folder.

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality of the toolkit.

## License

`qe_toolkit` is licensed under the GPL-3.0 License.
