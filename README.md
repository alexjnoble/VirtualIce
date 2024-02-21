# VirtualIce: Synthetic CryoEM Micrograph Generator

VirtualIce is a feature-rich synthetic cryoEM micrograph generator that uses buffer cryoEM micrographs with junk and carbon masked out as real background. It projects Protein Data Bank (PDB) structures onto buffer cryoEM micrographs, simulating realistic imaging conditions by adding noise and applying CTF to particles. It outputs particle coordinates after masking out junk. It outputs particles if requested.

#### Features

- Generates synthetic cryoEM micrographs and particles from buffer images and PDB IDs, EMDB IDs, or local files.
- Creates coordinate files (.star, .mod, .coord), not including particles obscured by junk/substrate.
- Adds Poisson and Gaussian noise to particles.
- Applies Contrast Transfer Function (CTF) to simulate microscope optics.
- Outputs micrographs as MRC, PNG, and JPEG output formats.
- Multi-core processing.
- Extensive customization options including particle distribution, ice thickness, and microscope parameters.

## Installation

VirtualIce requires Python 3, EMAN2, IMOD, and several dependencies, which can be installed using pip:

```bash
pip install numpy scipy matplotlib opencv-python-headless mrcfile
```

## Usage

The script can be run from the command line and takes a number of arguments.

Basic example usage:

```
./virtualice.py -s 1TIM -n 10
```

## Arguments

- `-s`, `--structures`: Specify PDB ID(s), EMDB ID(s), local files, and/or 'r' for random PDB/EMDB structures.
- `-n`, `--num_images`: Number of micrographs to generate.

Advanced example usage:

```
./virtualice.py -s 1TIM r my_structure.mrc 11638 -n 3 -I -P -J -Q 90 -b 4 -D n -p 2
```

## Arguments

- `-I`, `--imod_coordinate_file`: Also output one IMOD .mod coordinate file per micrograph.
- `-P`, `--png`: Output in PNG format.
- `-P`, `--jpeg`: Output in JPEG format.
- `-Q`, `--jpeg-quality`: JPEG image quality.
- `-b`, `--binning`: Bin micrographs by downsampling.
- `-D`, `--distribution`: Distribution type for generating particle locations.
- `-p`, `--parallel_processes`: Parallel processes for micrograph generation.

Additional arguments exist for fine-tuning the generation process including ice thickness, junk filtering, and CTF parameters.

## Ethical Use Agreement

VirtualIce is under the MIT License, allowing broad usage freedom, but we emphasize using it responsibly and ethically via these guidelines:

### Intended Use

VirtualIce is designed for educational and research purposes, specifically to aid in the development, testing, and validation of cryoEM image analysis algorithms by generating synthetic cryoEM micrographs and particles.

### Ethical Considerations

- **Transparency**: Any data generated using VirtualIce should be clearly marked as synthetic when published or shared, to distinguish it from real experimental data.
- **No Misrepresentation**: Users should not present synthetic data generated by VirtualIce as real data from physical experiments in any publications or presentations unless explicitly stated.
- **Research Integrity**: We encourage users to uphold the highest standards of scientific integrity in their work, ensuring that the use of synthetic data does not mislead, deceive, or otherwise harm the scientific community or the public.

### Embedded Metadata

To help enforce transparency, VirtualIce embeds metadata within each generated micrograph and particle file indicating that the data is synthetic. This metadata is designed to be tamper-evident, providing a means to verify the origin of the data.

## Issues and Support

If you encounter any problems or have any questions about the script, please [Submit an Issue](https://github.com/alexjnoble/VirtualIce/issues).

## Contributions

Contributions are welcome! Please open a [Pull Request](https://github.com/alexjnoble/VirtualIce/pulls) or [Issue](https://github.com/alexjnoble/VirtualIce/issues).

## Author

This script was written by Alex J. Noble with assistance from OpenAI's GPT-4 model, 2023-2024 at SEMC.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The Ethical Use Agreement is compatible with the MIT License, providing guidelines and recommendations for responsible use without legally restricting software use.
