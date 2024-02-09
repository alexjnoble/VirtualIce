# VirtualIce: Synthetic CryoEM Micrograph Generator

VirtualIce is a feature-rich synthetic cryoEM micrograph generator that uses buffer cryoEM micrographs with junk and carbon masked out as real background. It projects Protein Data Bank (PDB) structures onto buffer cryoEM micrographs, simulating realistic imaging conditions by adding noise and applying CTF to particles.

#### Features

- Generates synthetic cryoEM micrographs from PDB IDs or local PDB files.
- Supports MRC, PNG, and JPEG output formats.
- Incorporates realistic imaging conditions including Poisson and Gaussian noise.
- Applies Contrast Transfer Function (CTF) to simulate microscope optics.
- Multi-core processing for efficient image generation.
- Extensive customization options including particle distribution, ice thickness, and microscopy parameters.

## Installation

VirtualIce requires Python 3, EMAN2, IMOD, and several dependencies, which can be installed using pip:

```bash
pip install numpy scipy matplotlib opencv-python-headless mrcfile
```

## Usage

The script can be run from the command line and takes a number of arguments.

Example usage for generating synthetic micrographs:

```
./virtualice.py -p 1TIM -n 100 --png --jpeg -s 4
```

## Arguments

- `-p`, `--pdbs`: Specify PDB IDs or paths to local PDB files.
- `-n`, `--num_images`: Number of micrographs to generate.
- `--mrc`, `--png`, `--jpeg`: Output format options.
- `-s`, `--scale`: Scaling factor for particle images.
- `-a`, `--apix`: Angstrom per pixel for the output images.
- `-c`, `--cpus`: Number of CPU cores to use.

Additional arguments for fine-tuning the generation process including binning, noise simulation levels, and CTF application parameters.

## Issues and Support

If you encounter any problems or have any questions about the script, please [Submit an Issue](https://github.com/alexjnoble/VirtualIce/issues).

## Contributions

Contributions are welcome! Please open a [Pull Request](https://github.com/alexjnoble/VirtualIce/pulls) or [Issue](https://github.com/alexjnoble/VirtualIce/issues).

## Author

This script was written by Alex J. Noble with assistance from OpenAI's GPT-4 model, 2023-2024 at SEMC.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
