#!/usr/bin/env python3
#
# Author: Alex J. Noble, assisted by GPT, Claude, & Gemini, 2023-24 @SEMC, MIT License
#
# VirtualIce: Half-synthetic CryoEM Micrograph Generator
#
# This script generates half-synthetic cryoEM micrographs given protein structures and a list
# of noise micrographs and their defoci. It is intended that the noise micrographs are cryoEM
# images of buffer and that the junk & substrate are masked out using AnyLabeling.
#
# Dependencies: EMAN2 (namely e2pdb2mrc.py & e2proc3d.py)
#               pip install cupy gpustat mrcfile numpy opencv-python pandas scipy SimpleITK
#
# This program requires a separate installation of EMAN2 for proper functionality.
#
# EMAN2 is distributed under BSD-3-Clause & GPL-2.0 licenses. For details, see:
# - BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
# - GPL-2.0: https://opensource.org/licenses/GPL-2.0
# EMAN2 source code: https://github.com/cryoem/eman2
#
# IMOD (separate install; GPL-2.0 License) is optional to output IMOD coordinate files.
# IMOD source code & packages: https://bio3d.colorado.edu/imod/
#
# Ensure compliance with license terms when obtaining and using EMAN2 & IMOD.
__version__ = "1.0.0"

import os
import re
import cv2
import sys
import glob
import gzip
import json
import time
import random
import shutil
import gpustat
import inspect
import logging
import mrcfile
import argparse
import textwrap
import warnings
import itertools
import threading
import subprocess
import numpy as np
import pandas as pd
import SimpleITK as sitk
from multiprocessing import Pool
from urllib import request, error
from xml.etree import ElementTree as ET
from scipy.ndimage import affine_transform
from concurrent.futures import ProcessPoolExecutor
from scipy.fft import fft2, ifft2, fftshift, ifftshift
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage
    import cupyx.scipy.fftpack as cupy_fftpack
    from cupyx.scipy.ndimage import affine_transform as cp_affine_transform, gaussian_filter as cp_gaussian_filter
    gpu_available = True
except ImportError:
    import numpy as cp
    import scipy.ndimage as scipy_ndimage
    import scipy.fftpack as scipy_fftpack
    gpu_available = False

# Placeholder variables for the correct libraries to use
ndimage = None
fftpack = None

# Suppress warnings from mrcfile (filesize unexpected) & EMAN2 (smallest subnormal accuracy) imports
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mrcfile")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=(UserWarning))
    from EMAN2 import EMNumPy

# Global variable to store verbosity level
global_verbosity = 0

def check_num_particles(value):
    """
    Check if the number of particles is within the allowed range or 'max'.

    :param int/str value: Number of particles as an integer or 'max' as a string.
    :return int/str: Value if it is valid or 'max'.
    :raises ArgumentTypeError: If the value is not in the allowed range and not 'max'.
    """
    # Allow 'max' (case insensitive) to specify the maximum number of particles
    if str(value).lower() == 'max':
        return 'max'

    try:
        ivalue = int(value)
        if ivalue < 2 or ivalue >= 1000000:
            raise argparse.ArgumentTypeError("Number of particles must be between 2 and 1000000")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError("Number of particles must be an integer between 2 and 1000000 or 'max'")

def check_binning(value):
    """
    Check if the binning is within the allowed range.
    This function exists just so that ./virtualice.py -h doesn't blow up.

    :param int value: Binning requested.
    :return int: Value, if valid.
    :raises ArgumentTypeError: If the value is not in the allowed range.
    """
    ivalue = int(value)
    try:
        if ivalue < 2 or ivalue >= 64:
            raise argparse.ArgumentTypeError("Binning must be between 2 and 64")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError("Binning must be an integer between 2 and 64")

def parse_aggregation_amount(values, num_micrographs):
    """
    Parse the aggregation amount input by the user.

    The function accepts:
    - Numeric values between 0 and 10 (inclusive).
    - Strings of aggregation levels: 'low' or 'l', 'medium' or 'm', 'high' or 'h',
                                     and 'random' or 'r'.
    - Combinations of 'low', 'medium', and 'high' (e.g., 'low medium', 'l m').
    - Combinations of numbers (e.g. 2 5 6.7).
    - Combinations of strings and numbers (e.g. 'low 10').
    - 'none' (case insensitive) which maps to 0.
    - 'low medium high' or 'l m h' which maps to 'random'.
    - 'random' or 'r' to generate a list of random values.

    Depending on the input, it returns:
    - A list of float values representing the aggregation amounts.
    - 'low' or 'l': A random float between 0 and 3.3.
    - 'medium' or 'm': A random float between 3.3 and 6.7.
    - 'high' or 'h': A random float between 6.7 and 10.
    - Combinations like 'low medium' or 'r m h': One random float for each range.
    - 'random' or 'r': A random float between 0 and 10.
    - 'random random' or 'r r': A list of random floats between 0 and 10 of length of the
                                    number of micrographs requested.
    - Combinations ending in 'r' | 'random' like 'low 5 h r': A list of random floats in the
                                                              specified ranges/values of length of
                                                              the number of micrographs requested.

    :param list_of_single_valued_lists values: The input values for aggregation amount.
    :return list: The parsed aggregation amount as a list of floats.
    """
    print_and_log("", logging.DEBUG)
    def parse_single_value(value):
        value = value.lower()
        if value in ['none', 'n']:
            return [0]
        if value in ['low', 'l']:
            return [random.uniform(0, 3.3)]
        if value in ['medium', 'm']:
            return [random.uniform(3.3, 6.7)]
        if value in ['high', 'h']:
            return [random.uniform(6.7, 10)]
        if value in ['random', 'r']:
            return [random.uniform(0, 10)]
        numeric_value = float(value)
        # Apply floor and ceiling
        numeric_value = min(max(numeric_value, 0), 10)
        if 0 <= numeric_value <= 10:
            return [numeric_value]
        else:
            print_and_log(f"Warning: Input value {value} is outside the allowed range (0-10). Value adjusted to the nearest valid value.")
            return [numeric_value]

    if values[-1].lower() in {'r', 'random'}:
        # Handle the case where values is just 'r' or 'random'
        if len(values) == 1:
            random_value = random.uniform(0, 10)
            return [random_value] * num_micrographs  # Repeat the random value
        else:
            # Extract base values before 'random' or 'r'
            base_values = values[:-1]

            # Make a list of random selections from values the length of the number of micrographs requested
            combined_values = [parse_single_value(random.choice(base_values)) for _ in range(num_micrographs)]
            combined_values = [item for sublist in combined_values for item in sublist]

            return combined_values

    else:
        combined_values = []
        for value in values:
            combined_values.extend(parse_single_value(value))

        return combined_values

def validate_positive_int(parser, arg_name, value):
    """
    Validates if a given value is a positive integer.

    :param argparse.ArgumentParser parser: The argparse parser object.
    :param str arg_name: The name of the argument being validated.
    :param int value: The value to be validated.

    :raises argparse.ArgumentTypeError: If the value is not None and is <= 0.
    """
    print_and_log("", logging.DEBUG)
    try:
        if value is not None and value <= 0:
            parser.error(f"{arg_name} must be a positive integer.")
    except ValueError:
        raise argparse.ArgumentTypeError("{arg_name} must be a positive integer.")

def validate_positive_float(parser, arg_name, value):
    """
    Validates if a given value is a positive float.

    :param argparse.ArgumentParser parser: The argparse parser object.
    :param str arg_name: The name of the argument being validated.
    :param int value: The value to be validated.

    :raises argparse.ArgumentTypeError: If the value is not None and is <= 0.
    """
    print_and_log("", logging.DEBUG)
    try:
        if value is not None and value <= 0:
            parser.error(f"{arg_name} must be a positive float.")
    except ValueError:
        raise argparse.ArgumentTypeError("{arg_name} must be a positive float.")

def remove_duplicates_structures(lst):
    """
    Removes duplicate elements from the --structures list while converting to upper case
    for items without file extensions. Special entries like 'r', 'rp', 're', 'rm',
    'random' are not considered duplicates and can repeat.

    :param list lst: The input list from which duplicates need to be removed.

    :returns list: A new list with duplicate elements removed, preserving the order of the
                   first unique structure name (case-insensitive for items without file extensions).
    """
    print_and_log("", logging.DEBUG)
    seen = set()
    clean_lst = []
    duplicates = []
    exclude_duplicates = {'R', 'RP', 'RE', 'RM', 'RANDOM'}  # Set of entries to exclude from duplicate removal

    for item in lst:
        if '.' in item and item.rsplit('.', 1)[1]:  # Check if item has a file extension (ie. local file)
            normalized_item = item
        else:
            normalized_item = item.upper()

        if normalized_item in seen and normalized_item not in exclude_duplicates:
            duplicates.append(item)
        else:
            seen.add(normalized_item)
            clean_lst.append(item)

    if duplicates:
        print_and_log(f"\nRemoving duplicate {'request' if len(duplicates) == 1 else 'requests'}: {', '.join(duplicates)}\n")

    return clean_lst

def parse_arguments(script_start_time):
    """
    Parses command-line arguments.

    :param str script_start_time: Function start time, formatted as a string
    :returns argparse.Namespace: An object containing attributes for each command-line argument.
    """
    parser = argparse.ArgumentParser(description="\033[1mVirtualIce:\033[0m A feature-rich synthetic cryoEM micrograph generator that projects pdbs|mrcs onto existing buffer cryoEM micrographs. Star files for particle coordinates are outputed by default, mod and coord files are optional. Particle coordinates located within per-micrograph polygons at junk/substrate locations are projected but not written to coordinate files.",
    epilog="""
\033[1mExamples:\033[0m
  1. Basic usage: virtualice.py -s 1TIM -n 10
     Generates 10 random micrographs of PDB 1TIM.

  2. Advanced usage: virtualice.py -s 1TIM r my_structure.mrc 11638 rp -n 3 -I -P -J -Q 90 -b 4 -D n -ps 2 -C
     Generates 3 random micrographs of PDB 1TIM, a random EMDB/PDB structure, a local structure called my_structure.mrc, EMD-11638, and a random PDB.
     Outputs an IMOD .mod coordinate file, png, and jpeg (quality 90) for each micrograph, and bins all images by 4.
     Uses a non-random distribution of particles, parallelizes structure generation across 2 CPUs, and crops particles.

  3. Advanced usage: virtualice.py -s 1PMA -n 5 -om preferred -pw 0.9 -pa [*,90,0] [90 180 *] -aa l h r -ne --use_cpu -V 2 -3
     Generates 5 random micrographs of PDB 1PMA (proteasome) with preferred orientation for 90% of particles. The preferred orientations are defined
     by random selections of [*,0,90] (free to rotate along the first Z axis, then do not rotate in Y, then rotate 90 degrees in Z) and
     [90 180 0] (rotate 90 degrees along the first Z axis, then rotate 180 degrees in Y, then do not rotate in Z). The aggregation amount is
     randomly chosen from low and high values for each of the 5 micrographs. Edge particles are not included. Only CPUs are used (no GPUs).
     Terminal verbosity is set to 2. The resulting micrographs are opened with 3dmod after generation.
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter)  # Preserves whitespace for better formatting

    # Input Options
    input_group = parser.add_argument_group('\033[1mInput Options\033[0m')
    input_group.add_argument("-s", "--structures", type=str, nargs='+', default=['1TIM', '19436', 'r'], help="PDB ID(s), EMDB ID(s), names of local .pdb or .mrc/.map files, 'r' or 'random' for a random PDB or EMDB map, 'rp' for a random PDB, and/or 're' or 'rm' for a random EMDB map. Local .mrc/.map files must have voxel size in the header so that they are scaled properly. Separate structures with spaces. Note: PDB files are recommended because noise levels of .mrc/.map files are unpredictable. Default is %(default)s.")
    input_group.add_argument("-d", "--image_directory", type=str, default="ice_images", help="Local directory name where the micrographs are stored in mrc format. They need to be accompanied with a text file containing image names and defoci (see --image_list_file). Default directory is %(default)s")
    input_group.add_argument("-i", "--image_list_file", type=str, default="ice_images/good_images_with_defocus.txt", help="File containing local filenames of images with a defocus value after each filename (space between). Default is '%(default)s'.")
    input_group.add_argument("-me", "--max_emdb_size", type=float, default=512, help="The maximum allowed file size in megabytes. Default is %(default)s")

    # Particle and Micrograph Generation Options
    particle_micrograph_group = parser.add_argument_group('\033[1mParticle and Micrograph Generation Options\033[0m')
    particle_micrograph_group.add_argument("-n", "--num_images", type=int, default=5, help="Number of micrographs to create for each structure requested. Default is %(default)s")
    particle_micrograph_group.add_argument("-N", "--num_particles", type=check_num_particles, help="Number of particles to project onto the micrograph after rotation. Input an integer or 'max'. Default is a random number (weighted to favor numbers above 100 twice as much as below 100) up to a maximum of the number of particles that can fit into the micrograph without overlapping.")
    particle_micrograph_group.add_argument("-a", "--apix", type=float, default=1.096, help="Pixel size (in Angstroms) of the ice images, used to scale pdbs during pdb>mrc conversion (EMAN2 e2pdb2mrc.py option). Default is %(default)s (the pixel size of the ice images used during development)")
    particle_micrograph_group.add_argument("-r", "--pdb_to_mrc_resolution", type=float, default=3, help="Resolution in Angstroms for PDB to MRC conversion (EMAN2 e2pdb2mrc.py option). Default is %(default)s")
    particle_micrograph_group.add_argument("-st", "--std_threshold", type=float, default=-1.0, help="Threshold for removing noise from a downloaded/imported .mrc/.map file in terms of standard deviations above the mean. The idea is to not have dust around the 3D volume from the beginning. Default is %(default)s")
    particle_micrograph_group.add_argument("-nf", "--num_simulated_particle_frames", type=int, default=50, help="Number of simulated particle frames to generate Poisson noise and optionally apply dose damaging. Default is %(default)s")
    particle_micrograph_group.add_argument("-sp", "--scale_percent", type=float, default=33.33, help="How much larger to make the resulting mrc file from the pdb file compared to the minimum equilateral cube. Extra space allows for more delocalized CTF information (default: %(default)s; ie. %(default)s%% larger)")
    particle_micrograph_group.add_argument("-D", "--distribution", type=str, choices=['r', 'random', 'n', 'non_random', 'm', 'micrograph', 'g', 'gaussian', 'c', 'circular', 'ic', 'inverse_circular'], default='micrograph', help="Distribution type for generating particle locations: 'random' (or 'r') and 'non_random' (or 'n'). random is a random selection from a uniform 2D distribution. non_random selects from 4 distributions that can alternatively be requested directly: 1) 'micrograph' (or 'm') to mimic ice thickness (darker areas = more particles), 2) 'gaussian' (or 'g') clumps, 3) 'circular' (or 'c'), and 4) 'inverse_circular' (or 'ic'). Default is %(default)s which selects a distribution per micrograph based on internal weights.")
    particle_micrograph_group.add_argument("-aa", "--aggregation_amount", nargs='+', default=['low', 'random'], help="Amount of particle aggregation. Aggregation amounts can be set per-run or per-micrograph. To set per-run, input 0-10, 'low', 'medium', 'high', or 'random'. To set multiple aggregation amounts that will be chose randomly per-micrograph, input combinations like 'low medium', 'low high', '2 5', or 'low 3 9 10'. To set random aggregation amounts within a range, append any word input combination with 'random', like 'random random' to give the full range, or 'low medium random' to give a range from 0 to 6.7. Abbreviations work too, like '3.2 l h r'. Default is %(default)s")
    particle_micrograph_group.add_argument("-ao", "--allow_overlap", type=str, choices=['True', 'False', 'random'], default='random', help="Specify whether to allow overlapping particles. Options are 'True', 'False', or 'random'. Default is %(default)s")
    particle_micrograph_group.add_argument("-nl", "--num_particle_layers", type=int, default=2, help="If overlapping particles is allowed, this is the number of overlapping particle layers allowed (soft condition, not strict. Used in determining the maximum number of particles that can be placed in a micrograph). Default is %(default)s")
    particle_micrograph_group.add_argument("-so", "--save_overlapping_coords", action="store_true", help="Save overlapping particle coordinates to output files. Default is to not save overlapping particle")
    particle_micrograph_group.add_argument("-B", "--border", type=int, default=-1, help="Minimum distance of center of particles from the image border. Default is %(default)s = reverts to half boxsize")
    particle_micrograph_group.add_argument("-ne", "--no_edge_particles", action="store_true", help="Prevent particles from being placed up to the edge of the micrograph. By default, particles can be placed up to the edge.")
    particle_micrograph_group.add_argument("-se", "--save_edge_coordinates", action="store_true", help="Save particle coordinates that are closer than --border or closer than half a particle box size (if --border is not specified) from the edge. Requires --no_edge_particles to be False or --border to be greater than or equal to half the particle box size.")
    # TBD: Need to make a new border distance value for which partiles are saved based on distance from the borders
    #particle_micrograph_group.add_argument("-sb", "--save_border", type=int, default=None, help="Minimum distance from the image border required to save a particle's coordinates to the output files. Default is %(default)s, which will use the value of --border if specified, otherwise half of the particle box size.")

    # Simulation Options
    simulation_group = parser.add_argument_group('\033[1mSimulation Options\033[0m')
    simulation_group.add_argument("-dd", "--dose_damage", type=str, choices=['None', 'Light', 'Moderate', 'Heavy', 'Custom'], default='Moderate', help="Simulated protein damage due to accumulated dose, applied to simulated particle frames. Uses equation given by Grant & Grigorieff, 2015. Default is %(default)s")
    simulation_group.add_argument("-da", "--dose_a", type=float, required=False, help="Custom value for the \'a\' variable in equation (3) of Grant & Grigorieff, 2015 (only required if '--dose-preset Custom' is chosen).")
    simulation_group.add_argument("-db", "--dose_b", type=float, required=False, help="Custom value for the \'b\' variable in equation (3) of Grant & Grigorieff, 2015 (only required if '--dose-preset Custom' is chosen).")
    simulation_group.add_argument("-dc", "--dose_c", type=float, required=False, help="Custom value for the \'c\' variable in equation (3) of Grant & Grigorieff, 2015 (only required if '--dose-preset Custom' is chosen).")
    simulation_group.add_argument("-m", "--min_ice_thickness", type=float, default=30, help="Minimum ice thickness, which scales how much the particle is added to the image (this is a relative value). Default is %(default)s")
    simulation_group.add_argument("-M", "--max_ice_thickness", type=float, default=150, help="Maximum ice thickness, which scales how much the particle is added to the image (this is a relative value). Default is %(default)s")
    simulation_group.add_argument("-t", "--ice_thickness", type=float, help="Request a specific ice thickness, which scales how much the particle is added to the image (this is a relative value). This will override --min_ice_thickness and --max_ice_thickness. Note: When choosing 'micrograph' particle distribution, the ice thickness uses the same gradient map to locally scale simulated ice thickness.")
    simulation_group.add_argument("-ro", "--reorient_mrc", action="store_true", help="Reorient the MRC file (either provided as a .mrc file or requested from EMDB) so that the structure's principal axes align with the coordinate axes. Note: .pdb files are automatically reoriented, EMDB files are often too big to do so quickly. Default is %(default)s")
    simulation_group.add_argument("-om", "--orientation_mode", type=str, choices=['random', 'uniform', 'preferred'], default='random', help="Orientation mode for projections. Options are: random, uniform, preferred. Default is %(default)s")
    simulation_group.add_argument("-pa", "--preferred_angles", type=str, nargs='+', default=None, help="List of sets of three Euler angles (in degrees) for preferred orientations. Use '*' as a wildcard for random angles. Example: '[90,0,0]' or '[*,0,90]'. Euler angles are in the range [0, 360] for alpha and gamma, and [0, 180] for beta. Default is %(default)s")
    simulation_group.add_argument("-av", "--angle_variation", type=float, default=5.0, help="Standard deviation for normal distribution of variations around preferred angles (in degrees). Default is %(default)s")
    simulation_group.add_argument("-pw", "--preferred_weight", type=float, default=0.8, help="Weight of the preferred orientations in the range [0, 1] (only used if orientation_mode is preferred). Default is %(default)s")
    simulation_group.add_argument("-amp", "--ampcont", type=float, default=10, help="Amplitude contrast percentage when applying CTF to projections (EMAN2 CTF option). Default is %(default)s (ie. 10%%)")
    simulation_group.add_argument("-bf", "--bfactor", type=float, default=50, help="B-factor in A^2 when applying CTF to projections (EMAN2 CTF option). Default is %(default)s")
    simulation_group.add_argument("-cs", "--Cs", type=float, default=0.001, help="Microscope spherical aberration when applying CTF to projections (EMAN2 CTF option). Default is %(default)s because the microscope used to collect the provided buffer cryoEM micrographs has a Cs corrector")
    simulation_group.add_argument("-K", "--voltage", type=float, default=300, help="Microscope voltage (keV) when applying CTF to projections (EMAN2 CTF option). Default is %(default)s")

    # Junk Labels Options
    junk_labels_group = parser.add_argument_group('\033[1mJunk Labels Options\033[0m')
    junk_labels_group.add_argument("-nj", "--no_junk_filter", action="store_true", help="Turn off junk filtering; i.e. Do not remove particles from coordinate files that are on/near junk or substrate.")
    junk_labels_group.add_argument("-S", "--json_scale", type=int, default=4, help="Binning factor used when labeling junk to create the JSON files with AnyLabeling. Default is %(default)s")
    junk_labels_group.add_argument("-x", "--flip_x", action="store_true", help="Flip the polygons that identify junk along the x-axis")
    junk_labels_group.add_argument("-y", "--flip_y", action="store_true", help="Flip the polygons that identify junk along the y-axis")
    junk_labels_group.add_argument("-pe", "--polygon_expansion_distance", type=int, default=5, help="Number of pixels to expand each polygon in the JSON file that defines areas to not place particle coordinates. The size of the pixels used here is the same size as the pixels that the JSON file uses (ie. the binning used when labeling the micrographs in AnyLabeling). Default is %(default)s")

    # Particle Cropping Options
    particle_cropping_group = parser.add_argument_group('\033[1mParticle Cropping Options\033[0m')
    particle_cropping_group.add_argument("-C", "--crop_particles", action="store_true", help="Enable cropping of particles from micrographs. Particles will be extracted to the [structure_name]/Particles/ directory as .mrc files. Default is no cropping.")
    particle_cropping_group.add_argument("-CM", "--max_crop_particles", type=int, default=None, help="Maximum number of particles to crop from micrographs.")
    particle_cropping_group.add_argument("-X", "--box_size", type=int, default=None, help="Box size for cropped particles (x and y dimensions are the same). Particles with box sizes that fall outside the micrograph will not be cropped. Default is the size of the mrc used for particle projection after internal preprocessing.")

    # Micrograph and Coordinate Output Options
    output_group = parser.add_argument_group('\033[1mSystem and Program Options\033[0m')
    output_group.add_argument("-o", "--output_directory", type=str, help="Directory to save all output files. If not specified, a unique directory will be created.")
    output_group.add_argument("--mrc", action="store_true", default=True, help="Save micrographs as .mrc (default if no format is specified)")
    output_group.add_argument("-P", "--png", action="store_true", help="Save micrographs as .png")
    output_group.add_argument("-J", "--jpeg", action="store_true", help="Save micrographs as .jpeg")
    output_group.add_argument("-Q", "--jpeg_quality", type=int, default=95, help="Quality of saved .jpeg images (0 to 100). Default is %(default)s")
    output_group.add_argument("-b", "--binning", type=check_binning, default=1, help="Bin/Downsample the micrographs by Fourier cropping after superimposing particle projections. Binning is the sidelength divided by this factor (e.g. -b 4 for a 4k x 4k micrograph will result in a 1k x 1k micrograph) (e.g. -b 1 is unbinned). Default is %(default)s")
    output_group.add_argument("-k", "--keep", action="store_true", help="Keep the non-downsampled micrographs if downsampling is requested. Non-downsampled micrographs are deleted by default")
    output_group.add_argument("-I", "--imod_coordinate_file", action="store_true", help="Also output one IMOD .mod coordinate file per micrograph. Note: IMOD must be installed and working")
    output_group.add_argument("-O", "--coord_coordinate_file", action="store_true", help="Also output one .coord coordinate file per micrograph")
    output_group.add_argument("-3", "--view_in_3dmod", action='store_true', help="View generated micrographs in 3dmod at the end of the run")

    # System and Program Options
    misc_group = parser.add_argument_group('\033[1mSystem and Program Options\033[0m')
    misc_group.add_argument("--use_cpu", action='store_true', default=False, help="Use CPU for processing instead of GPU. Default: Use GPUs if available")
    misc_group.add_argument("-g", "--gpus", type=int, nargs='+', default=None, help="Specify which GPUs to use by their IDs for various processing steps: micrograph downsampling. Default: Use all available GPUs")
    misc_group.add_argument("-ps", "--parallelize_structures", type=int, default=None, help="Maximum number of parallel processes for each structure requested. Default is the number of structures requested or one-fourth the number of CPU cores available, whichever is smaller")
    misc_group.add_argument("-pm", "--parallelize_micrographs", type=int, default=None, help="Number of parallel processes for generating each micrograph. Default is the number of micrographs requested or one-fourth the number of CPU cores available, whichever is smaller")
    misc_group.add_argument("-c", "--cpus", type=int, default=None, help="Number of CPUs to use for various processing steps: Adding Poisson noise to and dose damaging particle frames, generating particle projections, micrograph downsampling, and particle cropping. Default is the number of CPU cores available, minus the number of structures parallelized across minus the number of micrographs parallelized across")
    misc_group.add_argument("-V", "--verbosity", type=int, default=1, help="Set verbosity level: 0 (quiet), 1 (some output), 2 (verbose), 3 (debug). For 0-2, a log file will be additionally written with 2. For 3, a log file will be additionally written with 3. Default is %(default)s")
    misc_group.add_argument("-q", "--quiet", action="store_true", help="Set verbosity to 0 (quiet). Overrides --verbosity if both are provided")
    misc_group.add_argument("-v", "--version", action="version", help="Show version number and exit", version=f"VirtualIce v{__version__}")

    args = parser.parse_args()

    # Set script start time variable
    args.script_start_time = script_start_time

    # Set verbosity level
    args.verbosity = 0 if args.quiet else args.verbosity

    # Make local paths absolute
    args.image_list_file = os.path.abspath(args.image_list_file)
    args.image_directory = os.path.abspath(args.image_directory)

    # Convert megabytes to bytes
    args.max_emdb_size = args.max_emdb_size * 1024 * 1024

    # Determine output directory
    if not args.output_directory:
        # Create a unique directory name using the date and time that the script was run
        args.output_directory = f"VirtualIce_run_{script_start_time}"

    # Create the output directory if it doesn't exist and move to it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    os.chdir(args.output_directory)

    # Setup logging based on the verbosity level
    setup_logging(script_start_time, args.verbosity)

    # Determine if GPU should be used
    args.use_gpu = not args.use_cpu and gpu_available

    # Set the appropriate ndimage and fftpack libraries based on user choice
    global ndimage, fftpack
    if args.use_gpu:
        ndimage = cupy_ndimage
        fftpack = cupy_fftpack
        args.gpu_ids = get_gpu_ids(args)
        if not args.use_gpu:  # Check if get_gpu_ids returned nothing; fallback to CPUs
            import numpy as cp
            import scipy.ndimage as scipy_ndimage
            import scipy.fftpack as scipy_fftpack
            ndimage = scipy_ndimage
            fftpack = scipy_fftpack
            args.gpu_ids = None
            print_and_log("GPUs or CuPy not configured properly. Falling back to CPUs for processing.")
        else:
            print_and_log(f"Using GPUs: {args.gpu_ids}", logging.DEBUG)
    elif gpu_available and not args.use_gpu:
        import numpy as cp
        import scipy.ndimage as scipy_ndimage
        import scipy.fftpack as scipy_fftpack
        ndimage = scipy_ndimage
        fftpack = scipy_fftpack
        args.gpu_ids = None
        print_and_log("Using only CPUs for processing.", logging.DEBUG)
    elif args.use_gpu and not gpu_available:
        ndimage = scipy_ndimage
        fftpack = scipy_fftpack
        args.gpu_ids = None
        print_and_log("GPUs requested, but not available or CuPy not installed. Using only CPUs for processing.")
    else:
        ndimage = scipy_ndimage
        fftpack = scipy_fftpack
        args.gpu_ids = None
        print_and_log("Using only CPUs for processing.", logging.DEBUG)

    # Automatically adjust parallelization settings based on available CPU cores
    available_cpus = os.cpu_count()
    if args.parallelize_structures == None:
        args.parallelize_structures = min(max(1, available_cpus // 4), len(args.structures))
    else:
        args.parallelize_structures = min(args.parallelize_structures, len(args.structures))
    if args.parallelize_micrographs == None:
        args.parallelize_micrographs = min(max(1, available_cpus // 4), args.num_images)
    else:
        args.parallelize_micrographs = min(args.parallelize_micrographs, args.num_images)
    if args.cpus == None:
        args.cpus = max(1, available_cpus - args.parallelize_structures - args.parallelize_micrographs)

    # Remove duplicate --structures
    args.structures = remove_duplicates_structures(args.structures)

    # Convert short particle distribution to full distribution name
    distribution_mapping = {'r': 'random', 'n': 'non_random', 'm': 'micrograph', 'g': 'gaussian', 'c': 'circular', 'ic': 'inverse_circular'}
    args.distribution_mapped = distribution_mapping.get(args.distribution, args.distribution)

    if args.crop_particles and not args.mrc:
        args.mrc = True
        print_and_log(f"Notice: Since cropping (--crop_particles) is requested, then --mrc must be turned on. --mrc is now set to True.")

    if not (args.mrc or args.png or args.jpeg):
        parser.error("No format specified for saving images. Please specify at least one format.")

    if not os.path.isfile(args.image_list_file):
        parser.error("The specified --image_list_file does not exist.")

    if not os.path.isdir(args.image_directory):
        parser.error("The specified --image_directory does not exist.")

    validate_positive_int(parser, "--num_images", args.num_images)
    validate_positive_float(parser, "--apix", args.apix)
    validate_positive_float(parser, "--pdb_to_mrc_resolution", args.pdb_to_mrc_resolution)
    validate_positive_float(parser, "--min_ice_thickness", args.min_ice_thickness)
    validate_positive_float(parser, "--max_ice_thickness", args.max_ice_thickness)
    validate_positive_float(parser, "--ice_thickness", args.ice_thickness)
    validate_positive_float(parser, "--ampcont", args.ampcont)
    validate_positive_float(parser, "--voltage", args.voltage)
    validate_positive_int(parser, "--json_scale", args.json_scale)
    validate_positive_int(parser, "--jpeg_quality", args.jpeg_quality)
    validate_positive_int(parser, "--parallelize_structures", args.parallelize_structures)

    # Dose damaging preset parameter mapping
    if args.dose_damage != 'Custom' and args.dose_damage != 'None':
        preset_values = {
            "Light": (0.245, -1.8, 12.01),
            "Moderate": (0.245, -1.665, 2.81),
            "Heavy": (0.245, -1.4, 2.01)
        }
        args.dose_a, args.dose_b, args.dose_c = preset_values[args.dose_damage]
    if args.dose_damage == 'Custom':
        if args.dose_a is None or args.dose_b is None or args.dose_c is None:
            parser.error("--dose_a, --dose_b, and --dose_c must be provided for --dose_damage=Custom.")
    if args.dose_damage == 'None':
        args.dose_a = args.dose_b = args.dose_c = 0
        args.num_simulated_particle_frames = 1  # No dose damage is equivalent to setting num_frames = 1 because it's just accumulating Poisson noise

    # Adjust allow_overlap to be random if not user-specified
    if args.allow_overlap == 'random':
        args.allow_overlap_random = True
    else:
        args.allow_overlap_random = False
        args.allow_overlap = (args.allow_overlap == 'True')

    # Print all arguments for the user's information
    formatted_output = ""
    for arg, value in vars(args).items():
        formatted_output += f"{arg}: {value};\n"
    formatted_output = formatted_output.rstrip(";\n")  # Remove the trailing semicolon
    argument_printout = textwrap.fill(formatted_output, width=80)  # Wrap the output text to fit in rows and columns

    print_and_log(f"\033[1m{'-' * 80}\n{('VirtualIce Run Configuration').center(80)}\n{'-' * 80}\033[0m", logging.WARNING)
    print_and_log(textwrap.fill(f"Generating {args.num_images} synthetic micrograph{'' if args.num_images == 1 else 's'} per structure ({', '.join(args.structures)}) using micrographs in {args.image_directory.rstrip('/')}/", width=80), logging.WARNING)
    print_and_log(f"\nInput command: {' '.join(sys.argv)}", logging.DEBUG)
    print_and_log("\nInput arguments:\n", logging.WARNING)
    print_and_log(argument_printout, logging.WARNING)
    print_and_log(f"\033[1m{'-' * 80}\033[0m\n", logging.WARNING)

    # This is put after the printout because it can create large lists depending on user input
    args.aggregation_amount = parse_aggregation_amount(args.aggregation_amount, args.num_images * len(args.structures))

    return args

def setup_logging(script_start_time, verbosity):
    """
    Sets up logging configuration for console and file output based on verbosity level.

    :param int script_start_time: Timestamp used for naming the log file.
    :param int verbosity: A value that determines the level of detail for log messages. Supports:
      - 0 for ERROR level messages only,
      - 1 for WARNING level and above,
      - 2 for INFO level and above,
      - 3 for DEBUG level and all messages, including detailed debug information.

    The function configures both a console handler and a file handler for logging,
    with messages formatted according to the specified verbosity level.
    """
    global global_verbosity
    global_verbosity = verbosity

    # Map verbosity to logging level
    levels = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}
    console_logging_level = levels.get(verbosity)

    log_filename = f"virtualice_{script_start_time}.log"

    # Formatters for logging
    simple_formatter = logging.Formatter('%(message)s')
    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d, %(funcName)s)')

    # File handler for logging, always records at least INFO level
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)  # Default to INFO level for file logging
    file_handler.setFormatter(detailed_formatter)

    # Console handler for logging, respects the verbosity level set by the user
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_logging_level)
    if verbosity < 3:
        console_handler.setFormatter(simple_formatter)
    else:
        console_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)  # Set file logging to DEBUG for verbosity 3

    # Configuring the logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set logger to highest level to handle all messages
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def print_and_log(message, level=logging.INFO):
    """
    Prints and logs a message with the specified level, including debug details for verbosity level 3.

    :param str message: The message to print and log.
    :param int level: The logging level for the message (e.g., logging.DEBUG).

    If verbosity is set to 3, the function logs additional details about the caller,
    including module name, function name, line number, and function parameters.

    This function writes logging information to the disk.
    """
    logger = logging.getLogger()

    if global_verbosity < 3:
        # Directly log the primary message with the specified level for verbosity less than 3
        logger.log(level, message)
    else:
        # Retrieve the caller's frame to get additional context for verbosity level 3
        caller_frame = inspect.currentframe().f_back
        func_name = caller_frame.f_code.co_name
        line_no = caller_frame.f_lineno
        module_name = caller_frame.f_globals["__name__"]

        # Skip logging debug information for print_and_log calls to avoid recursion
        if func_name != 'print_and_log':
            # Retrieve function parameters and their values for verbosity level 3
            args, _, _, values = inspect.getargvalues(caller_frame)
            args_info = ', '.join([f"{arg}={values[arg]}" for arg in args])

            # Construct the primary message with additional debug information
            detailed_message = f"{message} - Debug - Module: {module_name}, Function: {func_name}({args_info}), Line: {line_no}"
            logger.log(level, detailed_message)
        else:
            # For print_and_log function calls, log only the primary message
            logger.log(level, message)

def get_gpu_ids(args):
    """
    Determine which GPUs to use based on user input and GPU availability.

    :param argparse.Namespace args: Parsed command-line arguments.
    :return list_of_ints: List of GPU IDs to use.
    """
    print_and_log("", logging.DEBUG)
    if args.use_cpu or not gpu_available:
        return []
    if args.gpus is None:
        # Use all available GPUs (cp.cuda.runtime.getDeviceCount() breaks other cupy calls for some reason...)
        output = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, text=True)
        gpu_count = len(output.stdout.strip().split('\n'))
        if gpu_count == 0:
            args.use_gpu = False  # Switch to False if no GPUs are found
            return []
        return list(range(gpu_count))
    else:
        # Use only specified GPUs
        return args.gpus

def time_diff(time_diff):
    """
    Convert the time difference to a human-readable format.

    :param float time_diff: The time difference in seconds.
    :return str: A formatted string indicating the time difference.
    """
    print_and_log("", logging.DEBUG)
    seconds_in_day = 86400
    seconds_in_hour = 3600
    seconds_in_minute = 60

    days, time_diff = divmod(time_diff, seconds_in_day)
    hours, time_diff = divmod(time_diff, seconds_in_hour)
    minutes, seconds = divmod(time_diff, seconds_in_minute)

    time_str = ""
    if days > 0:
        time_str += f"{int(days)} day{'s' if days != 1 else ''}, "
    if hours > 0 or days > 0:  # Show hours if there are any days
        time_str += f"{int(hours)} hour{'s' if hours != 1 else ''}, "
    if minutes > 0 or hours > 0 or days > 0:  # Show minutes if there are any hours or days
        time_str += f"{int(minutes)} minute{'s' if minutes != 1 else ''}, "
    time_str += f"{int(seconds)} second{'s' if seconds != 1 else ''}"

    return time_str

def is_local_pdb_path(input_str):
    """
    Check if the input string is a path to a local PDB file.

    :param str input_str: The input string to be checked.
    :return bool: True if the input string is a valid local .pdb path, False otherwise.
    """
    print_and_log("", logging.DEBUG)
    full_path = os.path.join(os.pardir, input_str)
    return os.path.isfile(full_path) and input_str.endswith('.pdb')

def is_local_mrc_path(input_str):
    """
    Check if the input string is a path to a local MRC/Map file.

    :param str input_str: The input string to be checked.
    :return bool: True if the input string is a valid local .mrc or .map path, False otherwise.
    """
    print_and_log("", logging.DEBUG)
    full_path = os.path.join(os.pardir, input_str)
    return os.path.isfile(full_path) and (input_str.endswith('.mrc') or input_str.endswith('.map'))

def is_emdb_id(input_str):
    """
    Check if the input string structured as a valid EMDB ID.

    :param str input_str: The input string to be checked.
    :return bool: True if the input string is a valid EMDB ID format, False otherwise.
    """
    print_and_log("", logging.DEBUG)
    # EMDB ID must be 4 or 5 numbers
    return input_str.isdigit() and (len(input_str) == 4 or len(input_str) == 5)

def is_pdb_id(structure_input):
    """
    Check if the input string is a valid PDB ID.

    :param str structure_input: The input string to be checked.
    :return bool: True if the input string is a valid PDB ID, False otherwise.
    """
    print_and_log("", logging.DEBUG)
    # PDB ID must be 4 characters: first character is a number, next 3 are alphanumeric, & there must be at least one letter
    return bool(re.match(r'^[0-9][A-Za-z0-9]{3}$', structure_input) and any(char.isalpha() for char in structure_input))

def get_pdb_sample_name(pdb_id):
    """
    Retrieves the sample name for a given PDB ID from the RCSB database.

    :param str pdb_id: The PDB ID to query (e.g., '1ABC').
    :raises requests.HTTPError: If the request to the RCSB API fails.
    :raises json.JSONDecodeError: If the response from the RCSB API is not valid JSON.
    :returns str: The sample name associated with the PDB ID, or an error message.
    """
    print_and_log("", logging.DEBUG)
    url = f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}'
    try:
        with request.urlopen(url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                sample_name = data.get('struct', {}).get('title', 'Sample name not found')
                return sample_name.title()
            else:
                return 'Invalid PDB ID or network error'
    except Exception as e:
        return f'An error occurred: {e}'

def download_pdb(pdb_id, suppress_errors=False):
    """
    Download a PDB file from the RCSB website and print the sample name.
    Attempts to download the symmetrized .pdb1.gz file first.
    If the .pdb1.gz file is not available, download the regular PDB file.
    If both are available, get the larger one (sometimes entries have the symmetrized pdb as the regular file).

    :param str pdb_id: The ID of the PDB to be downloaded.
    :param bool suppress_errors: If True, suppress error messages. Useful for random PDB downloads.
    :return bool: True if the PDB exists, False if it doesn't.

    This function writes a .pdb file to the disk.
    """
    print_and_log("", logging.DEBUG)
    symmetrized_url = f"https://files.rcsb.org/download/{pdb_id}.pdb1.gz"
    regular_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    symmetrized_pdb_gz_path = f"{pdb_id}.pdb1.gz"
    symmetrized_pdb_path = f"{pdb_id}.pdb1"
    regular_pdb_path = f"{pdb_id}.pdb"
    downloaded_any = False

    if not suppress_errors:
        print_and_log(f"Downloading PDB {pdb_id}...")

    try:  # Try downloading the symmetrized .pdb1.gz file
        request.urlretrieve(symmetrized_url, symmetrized_pdb_gz_path)

        # Unzip the .pdb1.gz file
        with gzip.open(symmetrized_pdb_gz_path, 'rb') as f_in:
            with open(symmetrized_pdb_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the .pdb1.gz file
        os.remove(symmetrized_pdb_gz_path)
        downloaded_any = True
    except:
        pass

    try:  # Try downloading the regular .pdb file
        request.urlretrieve(regular_url, regular_pdb_path)
        downloaded_any = True
    except error.HTTPError as e:
        if not suppress_errors:
            print_and_log(f"Failed to download PDB {pdb_id}. HTTP Error: {e.code}\n", logging.WARNING)
        return False
    except Exception as e:
        if not suppress_errors:
            print_and_log(f"An unexpected error occurred while downloading PDB {pdb_id}. Error: {e}\n", logging.WARNING)
        return False

    if not downloaded_any:
        return False

    # Determine the largest .pdb file and remove the other one if both exist
    if os.path.exists(regular_pdb_path) and os.path.exists(symmetrized_pdb_path):
        if os.path.getsize(symmetrized_pdb_path) > os.path.getsize(regular_pdb_path):
            print_and_log(f"Downloaded symmetrized PDB {pdb_id}")
            os.remove(regular_pdb_path)
            os.rename(symmetrized_pdb_path, regular_pdb_path)
        else:
            print_and_log(f"Downloaded PDB {pdb_id}")
            os.remove(symmetrized_pdb_path)
    elif os.path.exists(symmetrized_pdb_path):
        os.rename(symmetrized_pdb_path, regular_pdb_path)

    return True

def download_random_pdb():
    """
    Download a random PDB file from the RCSB website.

    :return str: The ID of the PDB if downloaded successfully, otherwise False.
    """
    print_and_log("", logging.DEBUG)
    while True:
        valid_pdb_id = False
        while not valid_pdb_id:
            # Generate a random PDB ID starting with a number
            pdb_id = ''.join([random.choice('0123456789')] + random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=3))
            # Check if at least one of the last three characters is a letter to ensure the ID is not all numbers
            if any(char.isalpha() for char in pdb_id[1:]):
                valid_pdb_id = True

        # Attempt to download the PDB file
        success = download_pdb(pdb_id, suppress_errors=True)  # Suppress errors for random PDB download attempts
        if success:
            return pdb_id
        # No need to explicitly handle failure; loop repeats until a file downloads

def get_emdb_sample_name(emd_number):
    """
    Retrieves the name of the EMDB entry given its EMD number,
    prioritizing the 'sample' name and falling back to the 'title' name.

    :param str emd_number: The EMD number (e.g., '10025').
    :raises requests.HTTPError: If the request to the EMDB API fails.
    :raises xml.etree.ElementTree.ParseError: If XML file is not valid.
    :returns str: The name of the EMDB entry, or None if not found.
    """
    print_and_log("", logging.DEBUG)
    url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_number}/header/emd-{emd_number}-v30.xml"
    try:
        with request.urlopen(url) as response:
            if response.status == 200:
                root = ET.fromstring(response.read())
                sample_name = root.find('.//sample/name')
                if sample_name is not None:
                    return sample_name.text.title()
                else:
                    sample_name = root.find('.//title')
                    return sample_name.text.title() if sample_name is not None else None
            else:
                return None  # XML file not found or error
    except Exception as e:
        print_and_log(f"Error fetching XML data: {e}", logging.DEBUG)
        return None
    except request.HTTPError as e:
        print_and_log(f"HTTP Error fetching XML data: {e}", logging.DEBUG)
        return None

def download_emdb(emdb_id, max_emdb_size, suppress_errors=False):
    """
    Download and decompress an EMDB map file.

    :param str emdb_id: The ID of the EMDB map to be downloaded.
    :param int max_emdb_size: The maximum allowed file size in bytes.
    :param bool suppress_errors: If True, suppress error messages. Useful for random PDB downloads.
    :return bool: True if the map exists and is downloaded, False if not.

    This function writes a .map.gz file to the disk, then unzips the .map file and removes the .map.gz file.
    """
    print_and_log("", logging.DEBUG)
    url = f"https://files.wwpdb.org/pub/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz"
    local_filename = f"emd_{emdb_id}.map.gz"

    try:
        # Check the file size before downloading
        req = request.Request(url, method='HEAD')
        with request.urlopen(req) as response:
            file_size = int(response.headers.get('Content-Length', 0))
            if file_size > max_emdb_size:
                if not suppress_errors:
                    print_and_log(f"EMD-{emdb_id} file size ({file_size / (1024 * 1024):.2f} MB) exceeds the maximum allowed size ({max_emdb_size / (1024 * 1024):.2f} MB). Use the --max_emdb_size flag if you want to use this EMDB entry.", logging.WARNING)
                return False

        # Download the gzipped map file
        if not suppress_errors:
            print_and_log(f"Downloading EMD-{emdb_id}...")
        request.urlretrieve(url, local_filename)

        # Decompress the downloaded file
        with gzip.open(local_filename, 'rb') as f_in:
            with open(local_filename.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the compressed file after decompression
        os.remove(local_filename)

        print_and_log(f"Download and decompression complete for EMD-{emdb_id}.")
        sample_name = get_emdb_sample_name(emdb_id)
        print_and_log(f"[emd_{emdb_id}] Sample name: {sample_name}")
        return True
    except error.HTTPError as e:
        if not suppress_errors:
            print_and_log(f"EMD-{emdb_id} not found. HTTP Error: {e.code}", logging.WARNING)
        return False
    except Exception as e:
        if not suppress_errors:
            print_and_log(f"An unexpected error occurred while downloading EMD-{emdb_id}. Error: {e}", logging.WARNING)
        return False

def download_random_emdb(max_emdb_size):
    """
    Download a random EMDB map by trying random IDs.

    :param int max_emdb_size: The maximum allowed file size in bytes.
    :return str: The ID of the EMDB map if downloaded successfully, otherwise False.
    """
    print_and_log("", logging.DEBUG)
    while True:
        # Generate a random EMDB ID within a reasonable range
        emdb_id = str(random.randint(1, 45000)).zfill(4)  # Makes a 4 or 5 digit number with leading zeros. Random 1-3 digits will also be 4 digit.
        success = download_emdb(emdb_id, max_emdb_size, suppress_errors=True)
        if success:
            return emdb_id

def process_structure_input(structure_input, max_emdb_size, std_devs_above_mean, pixelsize):
    """
    Process each structure input by identifying whether it's a PDB ID for
    download, EMDB ID for download, a local file path, or a request for a random
    structure. Normalize any input .map/.mrc file and convert to .mrc.

    :param str structure_input: The structure input which could be a PDB ID, EMDB ID,
        a local file path, a request for a random PDB/EMDB structure ('r' or 'random'),
        a request for a random PDB structure ('rp'), a request for a random EMDB structure
        ('re' or 'rm').
    :param float std_devs_above_mean: Number of standard deviations above the mean to
        threshold downloaded/imported .mrc/.map files (for getting rid of some dust).
    :param float pixelsize: Pixel size of the micrograph onto which mrcs will be projected.
        Used to scale downloaded/imported .pdb/.mrc/.map files.
    :return tuple: A tuple containing the structure ID and file type if the file is
        successfully identified, downloaded, or a random structure is selected; None if
        there was an error or the download failed.
    """
    print_and_log("", logging.DEBUG)
    def process_local_mrc_file(file_path):
        converted_file = normalize_and_convert_mrc(file_path)
        threshold_mrc_file(f"{converted_file}.mrc", std_devs_above_mean)
        scale_mrc_file(f"{converted_file}.mrc", pixelsize)
        converted_file = normalize_and_convert_mrc(f"{converted_file}.mrc")
        return (converted_file, "mrc") if converted_file else None

    def download_random_pdb_structure():
        print_and_log("Downloading a random PDB...")
        pdb_id = download_random_pdb()
        return (pdb_id, "pdb") if pdb_id else None

    def download_random_emdb_structure(max_emdb_size):
        print_and_log("Downloading a random EMDB map...")
        emdb_id = download_random_emdb(max_emdb_size)
        structure_input = f"emd_{emdb_id}.map"
        return process_local_mrc_file(structure_input) if emdb_id else None

    if structure_input.lower() in ['r', 'random']:
        if random.choice(["pdb", "emdb"]) == "pdb":
            return download_random_pdb_structure()
        else:
            return download_random_emdb_structure(max_emdb_size)
    elif structure_input.lower() == 'rp':
        return download_random_pdb_structure()
    elif structure_input.lower() == 're' or structure_input.lower() == 'rm':
        return download_random_emdb_structure(max_emdb_size)
    elif is_local_pdb_path(structure_input):
        structure_name = os.path.basename(structure_input).split('.')[0]
        print_and_log(f"[{structure_name}] Using local PDB file: {structure_input}", logging.WARNING)
        # Make a local copy of the file
        full_path = os.path.join(os.pardir, structure_input)
        shutil.copy(full_path, os.path.basename(structure_input))
        return (os.path.basename(structure_input).split('.')[0], "pdb")
    elif is_local_mrc_path(structure_input):
        structure_name = os.path.basename(structure_input).split('.')[0]
        print_and_log(f"[{structure_name}] Using local MRC/MAP file: {structure_input}", logging.WARNING)
        # Make a local copy of the file
        full_path = os.path.join(os.pardir, structure_input)
        shutil.copy(full_path, os.path.basename(structure_input))
        return process_local_mrc_file(structure_input)
    elif is_emdb_id(structure_input):
        if download_emdb(structure_input, max_emdb_size):
            structure_input = f"emd_{structure_input}.map"
            return process_local_mrc_file(structure_input)
        else:
            return None
    elif is_pdb_id(structure_input):
        structure_name = structure_input
        if download_pdb(structure_input):
            print_and_log(f"[{structure_name}] Sample name: {get_pdb_sample_name(structure_input)}")
            return (structure_input, "pdb")
        else:
            print_and_log(f"Failed to download PDB: {structure_input}. Please check the ID and try again.", logging.WARNING)
            return None
    else:
        print_and_log(f"Unrecognized structure input: {structure_input}. Please enter a valid PDB ID, EMDB ID, local file path, or 'random'.", logging.WARNING)
        return None

def normalize_and_convert_mrc(input_file):
    """
    Normalize and, if necessary, pad the .map/.mrc file to make all dimensions equal,
    centering the original volume.

    This function normalizes the input MRC or MAP file using the `e2proc3d.py` script from
    EMAN2, ensuring the mean edge value is normalized. If the volume dimensions are not equal,
    it calculates the necessary padding to make the dimensions equal, with the original volume
    centered within the new dimensions. The adjusted volume is saved to the output file
    specified by the input file name or altered to have a '.mrc' extension if necessary.

    :param str input_file: Path to the input MRC or MAP file.
    :returns str: The base name of the output MRC file, without the '.mrc' extension, or None
        if an error occurred.

    This function modifies input_file if it's an .mrc file or writes an .mrc file and
    modifies it if input_file is a .map file.

    Note:
    - If the input file is not a cube (i.e., all dimensions are not equal), the function
    calculates the padding needed to center the volume within a cubic volume whose dimension
    is equal to the maximum dimension of the original volume.
    - The volume is padded with the average value found in the original data, ensuring that
    added regions do not introduce artificial density.
    - The function attempts to remove the original input file if it's different from the
    output file to avoid duplication.
    """
    print_and_log("", logging.DEBUG)
    with mrcfile.open(input_file, mode='r') as mrc:
        dims = mrc.data.shape
        max_dim = max(dims)

    # Calculate the padding needed to make all dimensions equal to the max dimension
    padding = [(max_dim - dim) // 2 for dim in dims]
    pad_width = [(pad, max_dim - dim - pad) for pad, dim in zip(padding, dims)]

    # Determine the output file name
    output_file = input_file if input_file.endswith('.mrc') else input_file.rsplit('.', 1)[0] + '.mrc'

    try:
        # Normalize the volume
        output = subprocess.run(["e2proc3d.py", input_file, output_file, "--outtype=mrc", "--process=normalize.edgemean"], capture_output=True, text=True, check=True)
        print_and_log(output.stdout, logging.DEBUG)

        # Read the normalized volume, pad it, and save to the output file
        with mrcfile.open(output_file, mode='r+') as mrc:
            data_padded = np.pad(mrc.data, pad_width, mode='constant', constant_values=np.mean(mrc.data))
            mrc.set_data(data_padded)
            mrc.update_header_from_data()
            mrc.close()

        # Remove the original input file if it's different from the output file
        if input_file != output_file and os.path.exists(input_file):
            os.remove(input_file)

    except subprocess.CalledProcessError as e:
        print_and_log(f"Error: {e.stderr}", logging.ERROR)
        return None

    return output_file.rsplit('.', 1)[0]

def threshold_mrc_file(input_file_path, std_devs_above_mean):
    """
    Thresholds an MRC file so that all voxel values below a specified number of 
    standard deviations above the mean are set to zero.

    :param str input_file_path: Path to the input MRC file.
    :param float std_devs_above_mean: Number of standard deviations above the mean for thresholding.
    :param str output_file_path: Path to the output MRC file. If None, overwrite the input file.

    This function modifies the input_file_path .mrc.
    """
    print_and_log("", logging.DEBUG)
    with mrcfile.open(input_file_path, mode='r+') as mrc:
        data = mrc.data
        mean = data.mean()
        std_dev = data.std()
        threshold = mean + (std_devs_above_mean * std_dev)
        data[data < threshold] = 0  # Set values below threshold to zero
        mrc.set_data(data)  # Update the MRC file with thresholded data

def scale_mrc_file(input_mrc_path, pixelsize):
    """
    Scale an MRC file to a specified pixel size, allowing both upscaling and downscaling.

    :param str input_mrc_path: Path to the input MRC file.
    :param float pixelsize: The desired pixel size in Angstroms.

    This function modifies the input_mrc_path .mrc.
    """
    print_and_log("", logging.DEBUG)
    # Read the current voxel size
    with mrcfile.open(input_mrc_path, mode='r') as mrc:
        original_voxel_size = mrc.voxel_size.x  # Assuming cubic voxels for simplicity
        original_shape = mrc.data.shape

    # Calculate the scale factor
    scale_factor = original_voxel_size / pixelsize

    # Calculate the new dimensions and round down to the next integer that is evenly divisible by 2 for future FFT processing
    scaled_dimension_x = int(((original_shape[0] * scale_factor) // 2) * 2) 
    scaled_dimension_y = int(((original_shape[1] * scale_factor) // 2) * 2)
    scaled_dimension_z = int(((original_shape[2] * scale_factor) // 2) * 2)

    # Constructs e2proc3d.py scaling command using temp file to avoid mrcfile warning from filesize/header mismatch
    if scale_factor < 1:
        command = ["e2proc3d.py",
            input_mrc_path, input_mrc_path,
            "--scale={}".format(scale_factor),
            "--clip={},{},{}".format(scaled_dimension_x, scaled_dimension_y, scaled_dimension_z)]
    elif scale_factor > 1:
        command = ["e2proc3d.py",
            input_mrc_path, input_mrc_path,
            "--clip={},{},{}".format(scaled_dimension_x, scaled_dimension_y, scaled_dimension_z),
            "--scale={}".format(scale_factor)]
    else:  # scale_factor == 1:
        return

    try:
        output = subprocess.run(command, capture_output=True, text=True, check=True)
        print_and_log(output, logging.DEBUG)
    except subprocess.CalledProcessError as e:
        print_and_log(f"Error during scaling operation: {e}", logging.WARNING)

def reorient_mrc(input_mrc_path):
    """
    Reorient an MRC file so that the structure's principal axes align with the coordinate axes.

    The function performs PCA on the non-zero voxels to align the longest axis with the x-axis,
    the second longest axis with the y-axis, and the third longest axis with the z-axis.

    :param str input_mrc_path: The path to the input MRC file.
    """
    print_and_log("", logging.DEBUG)
    data = readmrc(input_mrc_path)

    # Get the coordinates of non-zero voxels
    non_zero_indices = np.argwhere(data)

    # Perform PCA to find the principal axes
    mean_coords = np.mean(non_zero_indices, axis=0)
    centered_coords = non_zero_indices - mean_coords
    cov_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Align coordinates with principal axes
    aligned_coords = np.dot(centered_coords, sorted_eigenvectors)
    aligned_coords = np.round(aligned_coords).astype(int) + np.round(mean_coords).astype(int)

    # Create an empty array with the same shape as the original data
    reoriented_data = np.zeros_like(data)

    # Fill the reoriented data with the non-zero voxels
    for original_coord, new_coord in zip(non_zero_indices, aligned_coords):
        if 0 <= new_coord[0] < reoriented_data.shape[0] and 0 <= new_coord[1] < reoriented_data.shape[1] and 0 <= new_coord[2] < reoriented_data.shape[2]:
            reoriented_data[new_coord[0], new_coord[1], new_coord[2]] = data[original_coord[0], original_coord[1], original_coord[2]]

    return reoriented_data

def convert_pdb_to_mrc(pdb_name, apix, res):
    """
    Convert a PDB file to MRC format using EMAN2's e2pdb2mrc.py script.

    :param str pdb_name: The name of the PDB to be converted.
    :param float apix: The pixel size used in the conversion.
    :param int res: The resolution to be used in the conversion.

    :return int: The mass extracted from the e2pdb2mrc.py script output.

    This function writes a .pdb file to the disk.
    """
    print_and_log("", logging.DEBUG)
    print_and_log(f"[{pdb_name}] Converting PDB to MRC using EMAN2's e2pdb2mrc.py...")
    cmd = ["e2pdb2mrc.py", "--apix", str(apix), "--res", str(res), "--center", f"{pdb_name}.pdb", f"{pdb_name}.mrc"]
    output = subprocess.run(cmd, capture_output=True, text=True)
    print_and_log(output, logging.DEBUG)
    try:
        # Attempt to extract the mass from the output
        mass = int([line for line in output.stdout.split("\n") if "mass of" in line][0].split()[-2])
    except IndexError:
        # If the mass is not found in the output, set it to 0 and print a warning
        mass = 0
        print_and_log(f"[{pdb_name}]Warning: Mass not found for PDB. Setting mass to 0.", logging.WARNING)
    return mass

def readmrc(mrc_path):
    """
    Read an MRC file and return its data as a NumPy array.

    :param str mrc_path: The file path of the MRC file to read.
    :return numpy_array float: The data of the MRC file as a NumPy array
    """
    print_and_log("", logging.DEBUG)
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data
        numpy_array = np.array(data)

    return numpy_array

def writemrc(mrc_path, numpy_array, pixelsize=1.0):
    """
    Write a 2D or 3D NumPy array as an MRC file with specified pixel size.

    :param str mrc_path: The file path of the MRC file to write.
    :param numpy_array: The 2D or 3D NumPy array to be written.
    :param float pixelsize: The pixel/voxel size in Angstroms, assumed equal for all dimensions.
    :raises ValueError: If input numpy_array is not 2D or 3D.

    This function writes the numpy_array to the specified mrc_path with the pixel size in the header.
    """
    print_and_log("", logging.DEBUG)
    # Ensure the array is 2D or 3D
    if numpy_array.ndim not in [2, 3]:
        raise ValueError("Input array must be 2D or 3D")

    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(numpy_array)
        mrc.voxel_size = (pixelsize, pixelsize, pixelsize)
        mrc.update_header_from_data()
        mrc.update_header_stats()

    return

def lowPassFilter_gpu(imgarray_gpu, apix=1.0, bin=1, radius=0.0):
    """
    Low pass filter image to radius resolution using Gaussian smoothing on GPU.

    :param cupy.ndarray imgarray_gpu: The image data as a 2D or 3D CuPy array.
    :param float apix: The pixel size in Angstroms per pixel.
    :param int bin: The binning factor applied to the image.
    :param float radius: The desired resolution in Angstroms at which to apply the filter.
    :return cupy.ndarray: Filtered image array. Returns original array if radius is None or <= 0.
    """
    print_and_log("", logging.DEBUG)
    if radius is None or radius <= 0.0:
        return imgarray_gpu

    # Adjust sigma for apix and binning
    sigma = float(radius / apix / float(bin))

    # Apply Gaussian filter using CuPy
    filtered_imgarray_gpu = cp_gaussian_filter(imgarray_gpu, sigma=sigma / 10.0)  # 10.0 is a fudge factor that gets dose damaging about right.

    return filtered_imgarray_gpu

def lowPassFilter(imgarray, apix=1.0, bin=1, radius=0.0):
    """
    Low pass filter image to radius resolution using SimpleITK for Gaussian smoothing.

    The function smooths the image by applying a Gaussian filter, which is
    controlled by the specified radius. The effective width of the Gaussian
    filter is adjusted based on the pixel size (apix) and binning factor.

    :param numpy.ndarray imgarray: The image data as a 2D or 3D numpy array.
    :param float apix: The pixel size in Angstroms per pixel.
    :param int bin: The binning factor applied to the image.
    :param float radius: The desired resolution in Angstroms at which to apply the filter.
    :return numpy.ndarray: Filtered image array. Returns original array if radius is None or <= 0.
    """
    print_and_log("", logging.DEBUG)
    if radius is None or radius <= 0.0:
        return imgarray

    # Adjust sigma for apix and binning
    sigma = float(radius / apix / float(bin))

    # Convert numpy array to SimpleITK image for processing, then convert back
    sitkImage = sitk.GetImageFromArray(imgarray)
    sitkImage = sitk.SmoothingRecursiveGaussian(sitkImage, sigma=sigma / 10.0)  # 10.0 is a fudge factor that gets dose damaging about right.
    filtered_imgarray = sitk.GetArrayFromImage(sitkImage)

    return filtered_imgarray

def write_star_header(file_basename, apix, voltage, cs):
    """
    Write the header for a .star file.

    :param str file_basename: The basename of the file to which the header should be written.
    :param float apix: The pixel size used in the conversion.
    :param float voltage: The voltage used in the conversion.
    :param float cs: The spherical aberration used in the conversion.

    This function writes header information for a .star file to the disk with the specified parameters.
    """
    print_and_log("", logging.DEBUG)
    with open(f'{file_basename}.star', 'w') as star_file:
        star_file.write('\ndata_\n\n')
        star_file.write('loop_\n')
        star_file.write('_rlnVersion #1\n')
        star_file.write('3.1\n\n')  # Ensure compatibility with RELION 3.1 or newer
        star_file.write('data_optics\n')
        star_file.write('\n')
        star_file.write('loop_\n')
        star_file.write('_rlnOpticsGroupName #1\n')
        star_file.write('_rlnOpticsGroup #2\n')
        star_file.write('_rlnMicrographPixelSize #3\n')
        star_file.write('_rlnVoltage #4\n')
        star_file.write('_rlnSphericalAberration #5\n')
        star_file.write('_rlnAmplitudeContrast #6\n')
        star_file.write('_rlnImageDimensionality #7\n')
        star_file.write(f"opticsGroup1 1 {apix} {voltage} {cs} 0.1 2\n\n")
        star_file.write('data_particles\n')
        star_file.write('\n')
        star_file.write('loop_\n')
        star_file.write('_rlnMicrographName #1\n')
        star_file.write('_rlnCoordinateX #2\n')
        star_file.write('_rlnCoordinateY #3\n')
        star_file.write('_rlnAnglePsi #4\n')
        star_file.write('_rlnAngleRot #5\n')
        star_file.write('_rlnAngleTilt #6\n')
        star_file.write('_rlnOpticsGroup #7\n')
        star_file.write('_rlnDefocusU #8\n')
        star_file.write('_rlnDefocusV #9\n')
        star_file.write('_rlnDefocusAngle #10\n')

def write_all_coordinates_to_star(structure_name, image_path, particle_locations, orientations, defocus):
    """
    Write all particle locations and orientations to a STAR file.

    :param str structure_name: The name of the structure file.
    :param str image_path: The path of the image to add.
    :param list_of_tuples particle_locations: A list of tuples; each tuple contains the x, y coordinates.
    :param list_of_tuples orientations: List of orientations as tuples of Euler angles (alpha, beta, gamma) in degrees.
    :param float defocus: The defocus value to add to the STAR file.

    This function appends particle locations, orientations, and image_path to a .star file on the disk.
    """
    print_and_log("", logging.DEBUG)
    # Open the star file once and write all coordinates
    with open(f'{structure_name}.star', 'a') as star_file:
        for location, orientation in zip(particle_locations, orientations):
            x_shift, y_shift = location
            alpha, beta, gamma = orientation
            star_file.write(f'{image_path} {x_shift} {y_shift} {alpha} {beta} {gamma} 1 {defocus} {defocus} 0\n')

def convert_point_to_model(point_file, output_file):
    """
    Write an IMOD .mod file with particle coordinates.

    :param str point_file: Path to the input .point file.
    :param str output_file: Output file path for the .mod file.

    This function writes a .mod file to the output_file path.
    """
    print_and_log("", logging.DEBUG)
    try:
        # Run point2model command and give particles locations a circle of radius 3
        output = subprocess.run(["point2model", "-circle", "3", "-scat", point_file, output_file], capture_output=True, text=True, check=True)
        print_and_log(output, logging.DEBUG)
    except subprocess.CalledProcessError:
        print_and_log("Error while converting coordinates using point2model.", logging.WARNING)
    except FileNotFoundError:
        print_and_log("point2model not found. Ensure IMOD is installed and point2model is in your system's PATH.", logging.WARNING)

def write_mod_file(coordinates, output_file):
    """
    Write an IMOD .mod file with particle coordinates.

    :param list_of_tuples coordinates: List of (x, y) coordinates for the particles.
    :param str output_file: Output file path for the .mod file.

    This function converts particle coordinates in a .point file to an IMOD .mod file and writes it to output_file.
    """
    print_and_log("", logging.DEBUG)
    # Write the .point file
    point_file = os.path.splitext(output_file)[0] + ".point"
    with open(point_file, 'w') as f:
        for x, y in coordinates:
            f.write(f"{x} {y} 0\n")  # Writing each coordinate as a new line in the .point file

    # Convert the .point file to a .mod file
    convert_point_to_model(point_file, output_file)

def write_coord_file(coordinates, output_file):
    """
    Write a generic .coord file with particle (x,y) coordinates.

    :param list_of_tuples coordinates: List of (x, y) coordinates for the particles.
    :param str output_file: Output file path for the .coord file.

    This function writes a .coord to the output_file path.
    """
    print_and_log("", logging.DEBUG)
    coord_file = os.path.splitext(output_file)[0] + ".coord"
    with open(coord_file, 'w') as f:
        for x, y in coordinates:
            f.write(f"{x} {y}\n")  # Writing each coordinate as a new line in the .coord file

def save_particle_coordinates(structure_name, particle_locations_with_orientations, output_path, imod_coordinate_file, coord_coordinate_file, defocus):
    """
    Saves particle coordinates in specified formats (.star, .mod, .coord).

    :param str structure_name: Base name for the output files.
    :param list_of_tuples particle_locations_with_orientations: List of tuples where each tuple contains:
        - tuple 1: The (x, y) coordinates of the particle.
        - tuple 2: The (alpha, beta, gamma) Euler angles representing the orientation of the particle.
    :param list output_path: Output base filename.
    :param bool imod_coordinate_file: Whether to downsample and save IMOD .mod coordinate files.
    :param bool coord_coordinate_file: Whether to downsample and save .coord coordinate files.
    :param float defocus: The defocus value to add to the STAR file.

    This function writes a .star file and optionally a .mod and .coord file.
    """
    print_and_log("", logging.DEBUG)
    particle_locations = [loc for loc, ori in particle_locations_with_orientations]
    orientations = [ori for loc, ori in particle_locations_with_orientations]
    # Save .star file
    write_all_coordinates_to_star(structure_name, output_path + ".mrc", particle_locations, orientations, defocus)

    # Save IMOD .mod files
    if imod_coordinate_file:
        write_mod_file(particle_locations, os.path.splitext(output_path)[0] + ".mod")

    # Save .coord files
    if coord_coordinate_file:
        write_coord_file(particle_locations, os.path.splitext(output_path)[0] + ".coord")

def estimate_mass_from_map(mrc_name):
    """
    Estimate the mass of a protein from a cryoEM density map.

    :param str mrc_path: Path to the MRC/MAP file.
    :return float: Estimated mass of the protein in kilodaltons (kDa).

    This function estimates the mass of a protein based on the volume of density
    present in a cryoEM density map (MRC/MAP file) and the provided pixel size.
    It assumes an average protein density of 1.35 g/cm³ and uses the volume of
    voxels above a certain threshold to represent the protein. The threshold is set
    as the mean plus 2 standard deviation of the density values in the map. This
    is a simplistic thresholding approach and might need adjustment based on the
    specific map and protein.

    The estimated mass is returned in kilodaltons (kDa).

    Note: This method assumes the map is already thresholded appropriately and that
    the entire volume above the threshold corresponds to protein. In practice,
    determining an effective threshold can be challenging and may require manual
    intervention or advanced image analysis techniques.
    """
    print_and_log("", logging.DEBUG)
    protein_density_g_per_cm3 = 1.35  # Average density of protein
    angstroms_cubed_to_cm_cubed = 1e-24  # Conversion factor

    with mrcfile.open(f"{mrc_name}.mrc", mode='r') as mrc:
        data = mrc.data
        pixel_size_angstroms = mrc.voxel_size.x  # Assuming cubic voxels
        # 2 Standard deviations gave a reasonable fit for 10 random EMDB entries, but it's not very accurate or reliable.
        threshold = data.mean() + 2 * data.std()
        voxel_volume_angstroms_cubed = pixel_size_angstroms**3
        protein_volume_angstroms_cubed = np.sum(data > threshold) * voxel_volume_angstroms_cubed
        protein_volume_cm_cubed = protein_volume_angstroms_cubed * angstroms_cubed_to_cm_cubed

    mass_g = protein_volume_cm_cubed * protein_density_g_per_cm3
    mass_daltons = mass_g / 1.66053906660e-24  # Convert grams to daltons
    mass_kDa = mass_daltons / 1000  # Convert daltons to kilodaltons

    return mass_kDa

def read_polygons_from_json(json_file_path, expansion_distance, flip_x=False, flip_y=False, expand=True):
    """
    Read polygons from a JSON file generated by Anylabeling, optionally flip the coordinates,
    and optionally expand each polygon.

    :param str json_file_path: Path to the JSON file.
    :param int expansion_distance: Distance by which to expand the polygons.
    :param bool flip_x: Boolean to determine if the x-coordinates should be flipped.
    :param bool flip_y: Boolean to determine if the y-coordinates should be flipped.
    :param bool expand: Boolean to determine if the polygons should be expanded.
    :return list_of_tuples: List of polygons where each polygon is a list of (x, y) coordinates.
    """
    print_and_log("", logging.DEBUG)
    polygons = []

    # Read and parse the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

        image_width = data.get('imageWidth')
        image_height = data.get('imageHeight')

        # Extract and optionally flip polygons
        shapes = data.get('shapes', [])
        for shape in shapes:
            polygon = np.array(shape['points'])

            if flip_x:
                polygon[:, 0] = image_width - polygon[:, 0]

            if flip_y:
                polygon[:, 1] = image_height - polygon[:, 1]

            if expand:
                centroid = np.mean(polygon, axis=0)
                expanded_polygon = []
                for point in polygon:
                    vector = point - centroid  # Get a vector from centroid to point
                    unit_vector = vector / np.linalg.norm(vector)  # Normalize the vector
                    new_point = point + unit_vector * expansion_distance  # Move the point away from the centroid
                    expanded_polygon.append(new_point)
                polygon = np.array(expanded_polygon)

            polygons.append(polygon.tolist())

    return polygons

def filter_coordinates_outside_polygons(particle_locations, json_scale, polygons):
    """
    Filters out particle locations that are inside any polygon using OpenCV.

    :param list_of_tuples particle_locations: List of (x, y) coordinates of particle locations.
    :param int json_scale: Binning factor used when labeling junk to create the json file.
    :param list_of_tuples polygons: List of polygons where each polygon is a list of (x, y) coordinates.
    :return list_of_tuples: List of (x, y) coordinates of particle locations that are outside the polygons.
    """
    print_and_log("", logging.DEBUG)
    # An empty list to store particle locations that are outside the polygons
    filtered_particle_locations = []

    # Scale particle locations up to the proper image size
    scaled_particle_locations = [(int(x / json_scale), int(y / json_scale)) for x, y in particle_locations]

    # Iterate over each particle location
    for point in scaled_particle_locations:
        # Convert point to a numpy array
        point_array = np.array([point], dtype=np.int32)

        # Variable to keep track if a point is inside any polygon
        inside_any_polygon = False

        # Check each polygon to see if the point is inside
        for polygon in polygons:
            poly_np = np.array(polygon, dtype=np.int32)
            if cv2.pointPolygonTest(poly_np, point, False) >= 0:
                inside_any_polygon = True
                break  # Exit the loop if point is inside any polygon

        # If the point is not inside any polygon, add it to the filtered list
        if not inside_any_polygon:
            # Scale the point back to original scale
            filtered_particle_locations.append((point[0] * json_scale, point[1] * json_scale))

    return filtered_particle_locations

def extend_and_shuffle_image_list(num_images, image_list_file):
    """
    Extend (if necessary), shuffle, and select a specified number of random ice micrographs.

    :param int num_images: The number of images to select.
    :param str image_list_file: The path to the file containing the list of images.
    :return list: A list of selected ice micrograph filenames and defoci.
    """
    print_and_log("", logging.DEBUG)
    # Read the list of available micrographs and their defoci
    with open(image_list_file, "r") as f:
        image_list = [line.strip() for line in f.readlines() if line.strip()]

    if num_images <= len(image_list):
        # Shuffle the order of images randomly
        random.shuffle(image_list)
        # Select the desired number of images from the shuffled list and sort alphanumerically
        selected_images = sorted(image_list[:num_images])
    else:
        num_full_rounds = num_images // len(image_list)
        additional_images_needed = num_images % len(image_list)
        extended_list = image_list * num_full_rounds
        extended_list += random.sample(image_list, additional_images_needed)
        selected_images = sorted(extended_list)

    return selected_images

def non_uniform_random_number(min_val, max_val, threshold, weight):
    """
    Generate a non-uniform random number within a specified range.

    :param int min_value: The minimum value of the range (inclusive).
    :param int max_value: The maximum value of the range (inclusive).
    :param int threshold: The threshold value for applying the weighted value.
    :param float weighted_value: The weight assigned to values below the threshold.
    :return int: A non-uniform random number within the specified range.
    """
    print_and_log("", logging.DEBUG)
    # Create a population of numbers within the specified range
    population = range(min_val, max_val + 1)
    # Assign weights to each number, with 'weight' given to numbers below the threshold, and 1 for others
    weights = [weight if x < threshold else 1 for x in population]
    # Randomly select one number from the population based on the assigned weights
    chosen_number = random.choices(population, weights, k=1)[0]

    return int(chosen_number)

def next_divisible_by_primes(number, primes, count):
    """
    Find the next number divisible by a combination of primes.
    Useful for finding more optimal array shapes for FFT processing.

    :param int number: The starting number.
    :param list primes: A list of prime numbers to consider.
    :param int count: The number of primes to combine for finding the least common multiple.
    :return int: The smallest number divisible by the combination of primes.
    """
    print_and_log("", logging.DEBUG)
    # Store the least common multiples of prime combinations
    least_common_multiples = []

    # Iterate over all combinations of primes
    for prime_combination in itertools.combinations(primes, count):
        # Calculate the least common multiple of the combination
        least_common_multiple = np.lcm.reduce(prime_combination)
        # Find the next multiple of least_common_multiple greater than or equal to the starting number
        next_multiple = ((number + least_common_multiple - 1) // least_common_multiple) * least_common_multiple
        # Store the next multiple
        least_common_multiples.append(next_multiple)

    # Return the smallest number divisible by the combination of primes
    return min(least_common_multiples)

def get_max_batch_size(image_size, free_mem):
    """
    Determine the maximum batch size based on available GPU VRAM.

    :param int image_size: Size of a single image in bytes.
    :param int free_mem: Free memory available on the GPU.
    :return int: Maximum batch size that can fit in GPU memory.
    """
    print_and_log("", logging.DEBUG)
    # Leave some memory buffer (10% of total memory)
    buffer_mem = free_mem * 0.1
    available_mem = free_mem - buffer_mem

    # Calculate the maximum number of images that fit in the available memory
    max_batch_size = available_mem // image_size

    # Ensure at least one image can be processed
    return max(1, int(max_batch_size))

def get_gpu_utilization(gpu_id):
    """
    Get the current GPU utilization (both memory and core usage) for a specific GPU.

    :param int gpu_id: The ID of the GPU to query.
    :return dict: Dictionary containing free memory and core usage percentage.
    """
    print_and_log("", logging.DEBUG)
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu = gpu_stats[gpu_id]
    free_mem = gpu.memory_free
    core_usage = gpu.utilization
    return {'free_mem': free_mem, 'core_usage': core_usage}

def fourier_crop_gpu(image, downsample_factor):
    """	
    Fourier crops a 2D image using GPU.

    :param cupy.ndarray image: Input 2D image to be Fourier cropped.
    :param int downsample_factor: Factor by which to downsample the image in both dimensions.

    :return cupy.ndarray: Fourier cropped image.
    :raises ValueError: If input image is not 2D or if downsample factor is not valid.
    """
    print_and_log("", logging.DEBUG)
    image = cp.asarray(image)
    # Check if the input image is 2D
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")

    # Check that the downsampling factor is positive
    if downsample_factor <= 0 or not isinstance(downsample_factor, int):
        raise ValueError("Downsample factor must be a positive integer.")

    # Shift zero frequency component to center
    f_transform = cp.fft.fft2(image)
    f_transform_shifted = cp.fft.fftshift(f_transform)

    # Compute indices to crop the Fourier Transform
    center_x, center_y = cp.array(f_transform_shifted.shape) // 2
    crop_x_start = center_x - image.shape[0] // (2 * downsample_factor)
    crop_x_end = center_x + image.shape[0] // (2 * downsample_factor)
    crop_y_start = center_y - image.shape[1] // (2 * downsample_factor)
    crop_y_end = center_y + image.shape[1] // (2 * downsample_factor)

    # Crop the Fourier Transform
    f_transform_cropped = f_transform_shifted[crop_x_start:crop_x_end, crop_y_start:crop_y_end]

    # Inverse shift zero frequency component back to top-left
    f_transform_cropped_unshifted = cp.fft.ifftshift(f_transform_cropped)

    # Compute the Inverse Fourier Transform of the cropped Fourier Transform
    image_cropped = cp.fft.ifft2(f_transform_cropped_unshifted)

    # Take the real part of the result (to remove any imaginary components due to numerical errors) and move to a numpy array
    return cp.asnumpy(cp.real(image_cropped))

def fourier_crop(image, downsample_factor):
    """	
    Fourier crops a 2D image using CPU.

    :param numpy.ndarray image: Input 2D image to be Fourier cropped.
    :param int downsample_factor: Factor by which to downsample the image in both dimensions.

    :return numpy.ndarray: Fourier cropped image.
    :raises ValueError: If input image is not 2D or if downsample factor is not valid.
    """
    print_and_log("", logging.DEBUG)
    # Check if the input image is 2D
    if image.ndim != 2:
        raise ValueError("Input image must be 2D.")

    # Check that the downsampling factor is positive
    if downsample_factor <= 0 or not isinstance(downsample_factor, int):
        raise ValueError("Downsample factor must be a positive integer.")

    # Define the new shape; (x,y) pixel dimensions
    new_shape = (image.shape[0] // downsample_factor, image.shape[1] // downsample_factor)

    # Shift zero frequency component to center
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    # Compute indices to crop the Fourier Transform
    center_x, center_y = np.array(f_transform_shifted.shape) // 2
    crop_x_start = center_x - new_shape[0] // 2
    crop_x_end = center_x + new_shape[0] // 2
    crop_y_start = center_y - new_shape[1] // 2
    crop_y_end = center_y + new_shape[1] // 2

    # Crop the Fourier Transform
    f_transform_cropped = f_transform_shifted[crop_x_start:crop_x_end, crop_y_start:crop_y_end]

    # Inverse shift zero frequency component back to top-left
    f_transform_cropped_unshifted = ifftshift(f_transform_cropped)

    # Compute the Inverse Fourier Transform of the cropped Fourier Transform
    image_cropped = ifft2(f_transform_cropped_unshifted)

    # Take the real part of the result (to remove any imaginary components due to numerical errors)
    return np.real(image_cropped)

def downsample_micrograph(image_path, downsample_factor, pixelsize, use_gpu):
    """
    Downsample a micrograph by Fourier cropping and save it to a temporary directory.
    Supports mrc, png, and jpeg formats.

    :param str image_path: Path to the micrograph image file.
    :param int downsample_factor: Factor by which to downsample the image in both dimensions.
    :param float pixelsize: Pixel size of the micrograph.
    :param bool use_gpu: Whether to use GPU for processing.

    This function writes a .mrc/.png/.jpeg file to the disk.
    """
    print_and_log("", logging.DEBUG)
    try:
        # Determine the file format
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        if ext == '.mrc':
            image = readmrc(image_path)
        elif ext in ['.png', '.jpeg']:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply downsampling
        if use_gpu:
            downsampled_image = fourier_crop_gpu(image, downsample_factor)
        else:
            downsampled_image = fourier_crop(image, downsample_factor)

        # Save the downsampled micrograph
        bin_dir = os.path.join(os.path.dirname(image_path), f"bin_{downsample_factor}")
        os.makedirs(bin_dir, exist_ok=True)

        # Save the downsampled micrograph with the same name plus _bin## in the binned directory
        binned_image_path = os.path.join(bin_dir, f"{name}_bin{downsample_factor}{ext}")
        if ext == '.mrc':
            writemrc(binned_image_path, downsampled_image.astype(np.float32), downsample_factor * pixelsize)
        else:  # ext == .png/.jpeg
            # Normalize image to [0, 255] and convert to uint8
            downsampled_image -= downsampled_image.min()
            downsampled_image = downsampled_image / downsampled_image.max() * 255.0
            downsampled_image = downsampled_image.astype(np.uint8)
            cv2.imwrite(binned_image_path, downsampled_image)

    except Exception as e:
        print_and_log(f"Error processing {image_path}: {str(e)}")

def parallel_downsample_micrographs(image_directory, downsample_factor, pixelsize, cpus, use_gpu, gpu_ids):
    """
    Downsample all micrographs in a directory in parallel.

    :param str image_directory: Local micrograph directory name with .mrc/.png/.jpeg files.
    :param int downsample_factor: Factor by which to downsample the images in x,y.
    :param float pixelsize: Pixel size of the micrographs.
    :param int cpus: Number of CPUs to use if use_gpu is not True.
    :param bool use_gpu: Whether to use GPU for processing.
    :param list gpu_ids: List of GPU IDs to use for processing.
    """
    print_and_log("", logging.DEBUG)
    image_extensions = ['.mrc', '.png', '.jpeg']
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if os.path.splitext(filename)[1].lower() in image_extensions]

    if use_gpu:
        image_size = readmrc(next(file for file in image_paths if file.endswith('.mrc'))).nbytes
        batch_sizes = {}
        # Determine batch size for each GPU based on its available memory and utilization
        for gpu_id in gpu_ids:
            utilization = get_gpu_utilization(gpu_id)
            batch_size = get_max_batch_size(image_size, utilization['free_mem'])
            batch_sizes[gpu_id] = {'batch_size': batch_size, 'core_usage': utilization['core_usage']}

        # Distribute and process images
        start = 0
        while start < len(image_paths):
            # Sort GPUs by core usage in ascending order
            sorted_gpus = sorted(batch_sizes.items(), key=lambda x: x[1]['core_usage'])
            for gpu_id, batch_info in sorted_gpus:
                end = start + batch_info['batch_size']
                batch_paths = image_paths[start:end]
                if not batch_paths:
                    break
                with cp.cuda.Device(gpu_id):
                    for image_path in batch_paths:
                        downsample_micrograph(image_path, downsample_factor, pixelsize, use_gpu)
                start = end
                if start >= len(image_paths):
                    break
    else:
        # Downsample each micrograph by processing each image path in parallel
        pool = Pool(processes=cpus)
        pool.starmap(downsample_micrograph, [(image_path, downsample_factor, pixelsize, use_gpu) for image_path in image_paths])
        # Close the pool to prevent any more tasks from being submitted and wait for all worker processes to finish
        pool.close()
        pool.join()

def downsample_star_file(input_star, output_star, downsample_factor):
    """
    Read a STAR file, downsample the coordinates, and write a new STAR file.

    :param str input_star: Path to the input STAR file.
    :param str output_star: Path to the output STAR file.
    :param int downsample_factor: Factor by which to downsample the coordinates.

    This function writes a .star file to the disk.
    """
    print_and_log("", logging.DEBUG)
    with open(input_star, 'r') as infile, open(output_star, 'w') as outfile:
        # Flag to check if the line belongs to data_particles block
        in_data_particles_block = False
        for line in infile:
            # If line starts a data block, check if it is the data_particles block
            if line.startswith('data_'):
                in_data_particles_block = 'data_particles' in line

            # If in data_particles block and line contains coordinate data, modify it
            if in_data_particles_block and line.strip() and not line.startswith(('data_', 'loop_', '_')):
                parts = line.split()
                # Update micrograph name to include _bin##.mrc suffix
                parts[0] = f"{parts[0].replace('.mrc', '')}_bin{downsample_factor}.mrc"
                # Assuming coordinates are in the second and third column (index 1 and 2)
                parts[1] = str(float(parts[1]) / downsample_factor)  # Downsample x coordinate
                parts[2] = str(float(parts[2]) / downsample_factor)  # Downsample y coordinate
                # Join the line parts back together and write to the output file
                outfile.write(' '.join(parts) + '\n')
            else:
                # If line does not contain coordinate data, write it unchanged
                outfile.write(line)

def downsample_point_file(input_point, output_point, downsample_factor):
    """
    Read a .point file, downsample the coordinates, and write a new .point file.

    :param str input_point: Path to the input .point file.
    :param str output_point: Path to the output .point file.
    :param int downsample_factor: Factor by which to downsample the coordinates.

    This function writes a .point file to the disk.
    """
    print_and_log("", logging.DEBUG)
    with open(input_point, 'r') as infile, open(output_point, 'w') as outfile:
        for line in infile:
            # Skip empty lines
            if line.strip():
                # Split the line into x, y, and z coordinates
                x, y, z = map(float, line.split())
                # Downsample x and y coordinates
                x /= downsample_factor
                y /= downsample_factor
                # Write the downsampled coordinates to the output file
                outfile.write(f"{x:.2f} {y:.2f} {z:.2f}\n")

def downsample_coord_file(input_coord, output_coord, downsample_factor):
    """
    Read a .coord file, downsample the coordinates, and write a new .coord file.

    :param str input_coord: Path to the input .coord file.
    :param str output_coord: Path to the output .coord file.
    :param int downsample_factor: Factor by which to downsample the coordinates.

    This function writes a .coord file to the disk.
    """
    print_and_log("", logging.DEBUG)
    with open(input_coord, 'r') as infile, open(output_coord, 'w') as outfile:
        for line in infile:
            # Skip empty lines
            if line.strip():
                # Split the line into x and y coordinates
                x, y = map(float, line.split())
                # Downsample x and y coordinates
                x /= downsample_factor
                y /= downsample_factor
                # Write the downsampled coordinates to the output file
                outfile.write(f"{x:.2f} {y:.2f}\n")

def downsample_coordinate_files(structure_name, binning, imod_coordinate_file, coord_coordinate_file):
    """
    Downsample coordinate files based on the specified binning factor.

    :param str structure_name: Name of the structure.
    :param int binning: The factor by which to downsample the coordinates.
    :param bool imod_coordinate_file: Whether to downsample and save IMOD .mod coordinate files.
    :param bool coord_coordinate_file: Whether to downsample and save .coord coordinate files.
    """
    print_and_log("", logging.DEBUG)
    downsample_star_file(f"{structure_name}.star", f"{structure_name}_bin{binning}.star", binning)
    if imod_coordinate_file:
        for filename in os.listdir(f"{structure_name}/"):
            if filename.endswith(".point"):
                input_file = os.path.join(f"{structure_name}/", filename)
                output_point_file = os.path.join(f"{structure_name}/bin_{binning}/", filename.replace('.point', f'_bin{binning}.point'))
                # First downsample the .point files
                downsample_point_file(input_file, output_point_file, binning)
                # Then convert all of the .point files to .mod files
                mod_file = os.path.splitext(output_point_file)[0] + ".mod"
                convert_point_to_model(output_point_file, mod_file)
    if coord_coordinate_file:
        for filename in os.listdir(f"{structure_name}/"):
            if filename.endswith(".coord"):
                input_file = os.path.join(f"{structure_name}/", filename)
                output_coord_file = os.path.join(f"{structure_name}/bin_{binning}/", filename.replace('.coord', f'_bin{binning}.coord'))
                # First downsample the .coord files
                downsample_coord_file(input_file, output_coord_file, binning)

def read_star_particles(star_file_path):
    """
    Dynamically reads the 'data_particles' section from a STAR file and returns a DataFrame
    with particle coordinates and related data.

    :param str star_file_path: Path to the STAR file to read.
    :return dataframe: DataFrame with columns: micrograph names, particle coordinates, angles, optics group.
    :raises ValueError: If data_particles section of the STAR file is not found.
    """
    print_and_log("", logging.DEBUG)
    # Track the line number for where data begins
    data_start_line = None
    with open(star_file_path, 'r') as file:
        for i, line in enumerate(file):
            if 'data_particles' in line:
                # Found the data_particles section, now look for the actual data start
                data_start_line = i + 14  # Adjust if more lines are added to the star file
                break

    if data_start_line is None:
        raise ValueError("data_particles section not found in the STAR file.")

    # Read the data section from the identified start line, adjusting for the actual data start
    # Correct the `skiprows` approach to accurately target the start of data rows
    # Use `comment='#'` to ignore lines starting with '#'
    df = pd.read_csv(star_file_path, sep='\s+', skiprows=lambda x: x < data_start_line, header=None,
                     names=['micrograph_name', 'coord_x', 'coord_y', 'angle_psi', 'angle_rot', 'angle_tilt', 'optics_group', 'defocus_u', 'defocus_v', 'defocus_angle'], comment='#')

    return df

def trim_vol_determine_particle_numbers(mrc_array, input_micrograph, scale_percent, allow_overlap, num_particle_layers, num_particles):
    """
    Trim a volume and return a number of particles within a micrograph based on the maximum
    number of projections of this volume that can fit in the micrograph.

    :param numpy_array mrc_array: The input volume in MRC format.
    :param numpy_array input_micrograph: The input micrograph image (2D numpy array).
    :param float scale_percent: The percentage to scale the volume for trimming.
    :param bool allow_overlap: Flag to allow overlapping particles.
    :param int num_particle_layers: Number of layers of overlapping particles to project (only used if allow_overlap is True).
    :param int/str num_particles: Number of particles to project or 'max' for maximum particles.
    :return numpy_array, int, int: Trimmed MRC 3D array, number of particles to project, and the maximum number of particles.
    """
    print_and_log("", logging.DEBUG)

    # Find the non-zero entries and their indices
    non_zero_indices = np.argwhere(mrc_array)

    # Find the minimum and maximum indices for each dimension
    min_indices = np.min(non_zero_indices, axis=0)
    max_indices = np.max(non_zero_indices, axis=0) + 1

    # Compute the size of the largest possible equilateral cube
    min_cube_size = np.max(max_indices - min_indices)

    # Increase the cube size by the scale_percent
    cube_size = int(np.ceil(min_cube_size * (100 + scale_percent) / 100))

    # Find the next largest number that is divisible by at least 3 of the 5 smallest prime numbers
    primes = [2, 3, 5]
    cube_size = ((min(next_divisible_by_primes(cube_size, primes, 2), 336) + 1) // 2) * 2  # 336 is the largest practical box size before memory issues or seg faults

    # Adjust the minimum and maximum indices to fit the equilateral cube
    min_indices -= (cube_size - (max_indices - min_indices)) // 2
    max_indices = min_indices + cube_size

    # Handle boundary cases to avoid going beyond the original array size
    min_indices = np.maximum(min_indices, 0)
    max_indices = np.minimum(max_indices, mrc_array.shape)

    # Slice the original array to obtain the trimmed array
    trimmed_mrc_array = mrc_array[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]

    # Set the maximum number of particle projections that can fit in the image
    max_num_particles_without_overlap = int(2 * input_micrograph.shape[0] * input_micrograph.shape[1] / (trimmed_mrc_array.shape[0] * trimmed_mrc_array.shape[1]))

    # Determine max_num_particles based on whether overlap is allowed
    if allow_overlap:
        max_num_particles = num_particle_layers * max_num_particles_without_overlap
    else:
        max_num_particles = max_num_particles_without_overlap

    # Determine the number of particles to project based on user input or randomly
    if str(num_particles).lower() == 'max':
        num_particles_to_project = max_num_particles
    elif isinstance(num_particles, int):
        num_particles_to_project = min(num_particles, max_num_particles)
    else:
        # Choose a random number of particles between 2 and max_num_particles, with low particle numbers (<100) downweighted
        num_particles_to_project = non_uniform_random_number(2, max_num_particles, 100, 0.5)

    return trimmed_mrc_array, num_particles_to_project, max_num_particles

def determine_ice_and_particle_behavior(args, structure, micrograph, ice_scaling_fudge_factor, remaining_aggregation_amounts, context):
    """
    Determine ice and particle behaviors: First trim the volume and determine possible numbers of particles,
    then determine ice thickness, then determine particle distribution, then determine particle aggregation.

    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :param numpy_array structure: 3D numpy array of the structure for which to generate projections.
    :param numpy_array micrograph: 2D numpy array representing the ice micrograph.
    :param float ice_scaling_fudge_factor: Fudge factor for making particles look dark enough.
    :param list remaining_aggregation_amounts: Keep track of aggregation amounts to not repeat them.
    :param str context: Context string for print statements (structure name and micrograph number).
    :return tuple: ice_thickness, ice_thickness_printout, num_particles, dist_type, non_random_dist_type, aggregation_amount_val
    """
    print_and_log("", logging.DEBUG)
    print_and_log(f"{context} Trimming the mrc...")
    # Set `num_particles` based on the user input (args.num_particles) with the following rules:
    # 1. If the user provides a value for `args.num_particles` and it is <= the `max_num_particles`, use it.
    # 2. If the user does not provide a value, use `rand_num_particles`.
    # 3. If the user's provided value exceeds `max_num_particles`, use `max_num_particles` instead.
    # 4. If the user specifies 'max', use max_num_particles, otherwise apply the existing conditions

    structure, rand_num_particles, max_num_particles = trim_vol_determine_particle_numbers(
        structure, micrograph, args.scale_percent, args.allow_overlap, args.num_particle_layers, args.num_particles)
    
    num_particles = (max_num_particles if str(args.num_particles).lower() == 'max' else
             args.num_particles if args.num_particles and isinstance(args.num_particles, int) and args.num_particles <= max_num_particles else
             rand_num_particles if not args.num_particles else
             max_num_particles)

    if args.num_particles:
        print_and_log(f"{context} Attempting to find {int(num_particles)} particle locations in the micrograph...")
    else:
        print_and_log(f"{context} Choosing a random number of particles to add to the micrograph...")

    # Random or user-specified ice thickness
    # Adjust the relative ice thickness to work mathematically
    # (yes, the inputs are inversely named and there are fudge factors just so the user gets a number that feels right)
    if args.ice_thickness is None:
        min_ice_thickness = ice_scaling_fudge_factor/(0.32*args.max_ice_thickness + 20.45)
        max_ice_thickness = ice_scaling_fudge_factor/(0.32*args.min_ice_thickness + 20.45)
        ice_thickness = random.uniform(min_ice_thickness, max_ice_thickness)
    else:
        ice_thickness = ice_scaling_fudge_factor / (0.32*args.ice_thickness + 20.45)
    ice_thickness_printout = (ice_scaling_fudge_factor - 20.45*ice_thickness)/(0.32*ice_thickness)

    # Set the particle distribution type. Default to 'micrograph'
    non_random_distributions = {'micrograph', 'gaussian', 'circular', 'inverse_circular'}
    if args.distribution_mapped in non_random_distributions:
        dist_type = 'non_random'
        non_random_dist_type = args.distribution_mapped
    elif args.distribution_mapped == 'random':
        dist_type = 'random'
        non_random_dist_type = None
    elif args.distribution_mapped == 'non_random':
        dist_type = 'non_random'
        # Randomly select a non-random distribution, weighted towards micrograph because it is the most realistic.
        # Note: gaussian can create 1-5 gaussian blobs on the micrograph
        non_random_dist_type = np.random.choice(['circular', 'inverse_circular', 'gaussian', 'micrograph'], p=[0.0025, 0.0075, 0.19, 0.8])
    else:  # args.distribution_mapped == None
        dist_type = 'non_random'
        non_random_dist_type = 'micrograph'
    if dist_type == 'non_random':
        if not args.num_particles:
            if non_random_dist_type == 'circular':
                # Reduce the number of particles because the maximum was calculated based on the maximum number of particles that will fit in the micrograph side-by-side
                num_particles = max(num_particles // 2, 2)  # Minimum number of particles is 2
            elif non_random_dist_type == 'inverse_circular':
                # Reduce the number of particles because the maximum was calculated based on the maximum number of particles that will fit in the micrograph side-by-side
                num_particles = max(num_particles * 2 // 3, 2)
            elif non_random_dist_type == 'gaussian':
                # Reduce the number of particles because the maximum was calculated based on the maximum number of particles that will fit in the micrograph side-by-side
                num_particles = max(num_particles // 4, 2)

    print_and_log(f"{context} {num_particles} particles will be added to the micrograph")

    if args.aggregation_amount:
        if not remaining_aggregation_amounts:
            remaining_aggregation_amounts = list(args.aggregation_amount)
        aggregation_amount_index = random.randint(0, len(remaining_aggregation_amounts) - 1)
        aggregation_amount_val = remaining_aggregation_amounts.pop(aggregation_amount_index)
    else:
        aggregation_amount_val = 0

    return ice_thickness, ice_thickness_printout, num_particles, dist_type, non_random_dist_type, aggregation_amount_val

def parse_preferred_angles(preferred_angles):
    """
    Parse a list of preferred angles into a structured format.
    This function allows the user to input angles in any combination of these formats:
    [0,180,90] [0 180 90] [0, 180, 90]

    :param list preferred_angles: List of strings representing preferred angles.
    :return list_of_tuples: List of preferred angles in numeric format.
    """
    print_and_log("", logging.DEBUG)
    parsed_angles = []
    current_angle_set = []

    for item in preferred_angles:
        if item.startswith('[') and item.endswith(']'):
            # Handle complete angle sets like '[0,180,0]'
            angles = item.strip('[]').split(',')
            parsed_angles.append(tuple(angle.strip() for angle in angles))
        elif item.startswith('['):
            if current_angle_set:
                parsed_angles.append(tuple(current_angle_set))
                current_angle_set = []
            current_angle_set.append(item.lstrip('['))
        elif item.endswith(']'):
            current_angle_set.append(item.rstrip(']'))
            parsed_angles.append(tuple(current_angle_set))
            current_angle_set = []
        else:
            current_angle_set.append(item)

    if current_angle_set:
        parsed_angles.append(tuple(current_angle_set))

    # Clean up each angle in the sets
    cleaned_angles = []
    for angle_set in parsed_angles:
        cleaned_set = []
        for angle in angle_set:
            cleaned_angle = re.sub(r'[,\s]+', '', angle).strip()
            if cleaned_angle:
                cleaned_set.append(cleaned_angle)
        cleaned_angles.append(tuple(cleaned_set))

    return cleaned_angles

def generate_orientations(preferred_angles, angle_variation, num_preferred, num_random):
    """
    Generate a list of orientations based on preferred angles with specified variations
    and a number of random orientations.

    :param list_of_tuples preferred_angles: List of preferred angles in numeric format.
    :param float angle_variation: Standard deviation for the normal distribution of angle variations.
    :param int num_preferred: Number of preferred orientations to generate.
    :param int num_random: Number of random orientations to generate.
    :return list_of_tuples: List of orientations with preferred and random angles.
    """
    print_and_log("", logging.DEBUG)
    preferred_orientations = []
    for _ in range(num_preferred):
        angles = preferred_angles[np.random.randint(len(preferred_angles))]
        alpha, beta, gamma = angles
        alpha = np.round(np.random.uniform(0, 360), 3) if alpha == '*' else np.round(np.random.normal(float(alpha), angle_variation / 2), 3)  # angle_variation / 2 because this is +-angle_variation
        beta = np.round(np.random.uniform(0, 180), 3) if beta == '*' else np.round(np.random.normal(float(beta), angle_variation / 2), 3)
        gamma = np.round(np.random.uniform(0, 360), 3) if gamma == '*' else np.round(np.random.normal(float(gamma), angle_variation / 2), 3)
        preferred_orientations.append((alpha % 360, beta % 180, gamma % 360))

    random_orientations = [(np.round(np.random.uniform(0, 360), 3), np.round(np.random.uniform(0, 180), 3), np.round(np.random.uniform(0, 360), 3)) for _ in range(num_random)]
    return preferred_orientations + random_orientations

def euler_to_matrix_gpu(alpha, beta, gamma):
    """
    Convert ZYZ Euler angles to a rotation matrix using CuPy.

    :param float alpha: First rotation angle in degrees.
    :param float beta: Second rotation angle in degrees.
    :param float gamma: Third rotation angle in degrees.
    :return cupy.ndarray: The combined rotation matrix.
    """
    print_and_log("", logging.DEBUG)
    alpha = cp.deg2rad(cp.asarray(alpha))
    beta = cp.deg2rad(cp.asarray(beta))
    gamma = cp.deg2rad(cp.asarray(gamma))

    cos_alpha, sin_alpha = cp.cos(alpha), cp.sin(alpha)
    cos_beta, sin_beta = cp.cos(beta), cp.sin(beta)
    cos_gamma, sin_gamma = cp.cos(gamma), cp.sin(gamma)

    Rz1 = cp.zeros((3, 3), dtype=cp.float32)
    Rz1[0, 0] = cos_alpha
    Rz1[0, 1] = -sin_alpha
    Rz1[1, 0] = sin_alpha
    Rz1[1, 1] = cos_alpha
    Rz1[2, 2] = 1

    Ry = cp.zeros((3, 3), dtype=cp.float32)
    Ry[0, 0] = cos_beta
    Ry[0, 2] = sin_beta
    Ry[1, 1] = 1
    Ry[2, 0] = -sin_beta
    Ry[2, 2] = cos_beta

    Rz2 = cp.zeros((3, 3), dtype=cp.float32)
    Rz2[0, 0] = cos_gamma
    Rz2[0, 1] = -sin_gamma
    Rz2[1, 0] = sin_gamma
    Rz2[1, 1] = cos_gamma
    Rz2[2, 2] = 1

    R = cp.dot(Rz2, cp.dot(Ry, Rz1))
    return R

def euler_to_matrix(alpha, beta, gamma):
    """
    Convert ZYZ Euler angles to a rotation matrix.

    :param float alpha: First rotation angle in degrees.
    :param float beta: Second rotation angle in degrees.
    :param float gamma: Third rotation angle in degrees.
    :return numpy.ndarray: The combined rotation matrix.
    """
    print_and_log("", logging.DEBUG)
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    Rz1 = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz2 = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])

    R = np.dot(Rz2, np.dot(Ry, Rz1))
    return R

def trim_volume(volume_data):
    """
    Trim the volume data to the smallest cube containing non-zero voxels.

    :param numpy.ndarray volume_data: 3D volume data to be trimmed.
    :return numpy.ndarray: Trimmed 3D volume data.
    """
    print_and_log("", logging.DEBUG)
    # Find the non-zero entries and their indices
    non_zero_indices = np.argwhere(volume_data)

    # Calculate the centroid of the non-zero voxels
    centroid = np.mean(non_zero_indices, axis=0)

    # Calculate the distances from the centroid to all non-zero voxels
    distances = np.linalg.norm(non_zero_indices - centroid, axis=1)

    # Find the maximum distance (radius) and compute the cube size needed
    max_distance = np.max(distances)
    min_cube_size = int(np.ceil(2 * max_distance))

    # Calculate the start and end indices for the trimmed volume
    center_idx = centroid.astype(int)
    half_size = min_cube_size // 2
    start_idx = np.maximum(center_idx - half_size, 0)
    end_idx = np.minimum(center_idx + half_size, volume_data.shape)

    # Slice the original array to obtain the trimmed array
    trimmed_volume = volume_data[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]

    return trimmed_volume

def pad_projection(projection, original_shape):
    """
    Pad the 2D projection back to the original shape with zeros.

    :param numpy.ndarray projection: 2D projection to be padded.
    :param tuple original_shape: Original shape of the 3D volume.
    :return numpy.ndarray: Padded 2D projection.
    """
    print_and_log("", logging.DEBUG)
    pad_x = (original_shape[0] - projection.shape[0]) // 2
    pad_y = (original_shape[1] - projection.shape[1]) // 2

    padded_projection = np.pad(projection, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')

    return padded_projection

def generate_projection_gpu(angle, volume_data_gpu, original_shape):
    """
    Generate a 2D projection of the 3D volume data on the GPU by rotating it to the specified angle using the ZYZ convention.

    :param tuple_of_floats angle: Tuple of three Euler angles (alpha, beta, gamma) in degrees.
    :param cupy.ndarray volume_data_gpu: 3D volume data on GPU to be projected.
    :param tuple original_shape: Original shape of the 3D volume.
    :return numpy.ndarray: 2D projection of the volume data.
    """
    print_and_log("", logging.DEBUG)
    alpha, beta, gamma = angle
    rotation_matrix = euler_to_matrix_gpu(alpha, beta, gamma)

    # Center of the trimmed volume
    center = cp.array(volume_data_gpu.shape) / 2

    # Apply affine transformation
    rotated_volume_gpu = cp_affine_transform(volume_data_gpu, rotation_matrix, offset=center - cp.dot(rotation_matrix, center), order=1)

    # Project the rotated volume
    projection_gpu = cp.sum(rotated_volume_gpu, axis=2)

    # Pad the projection back to the original shape
    padded_projection = pad_projection(cp.asnumpy(projection_gpu), original_shape)

    return padded_projection

def generate_projection(angle, volume_data):
    """
    Generate a 2D projection of the 3D volume data by rotating it to the specified angle using the ZYZ convention.

    :param tuple_of_floats angle: Tuple of three Euler angles (alpha, beta, gamma) in degrees.
    :param numpy.ndarray volume_data: 3D volume data to be projected.
    :return numpy.ndarray: 2D projection of the volume data.
    """
    print_and_log("", logging.DEBUG)
    alpha, beta, gamma = angle
    rotation_matrix = euler_to_matrix(alpha, beta, gamma)

    # Trim the volume to the smallest possible cube containing non-zero voxels
    trimmed_volume = trim_volume(volume_data)

    # Center of the trimmed volume
    center = np.array(trimmed_volume.shape) / 2

    # Apply affine transformation
    rotated_volume = affine_transform(trimmed_volume, rotation_matrix, offset=center - np.dot(rotation_matrix, center), order=1)

    # Project the rotated volume
    projection = np.sum(rotated_volume, axis=2)

    # Pad the projection back to the original shape
    original_shape = volume_data.shape[:2]
    padded_projection = pad_projection(projection, original_shape)

    return padded_projection

def generate_projections(structure, num_projections, orientation_mode, preferred_angles, angle_variation, preferred_weight, num_cores, use_gpu, gpu_ids):
    """
    Generate a list of projections for a given structure based on the specified orientation mode.

    :param numpy_array structure: 3D numpy array of the structure for which to generate projections.
    :param int num_projections: Number of projections to generate.
    :param str orientation_mode: Orientation mode for generating projections ('random', 'uniform', 'preferred').
    :param list_of_strs preferred_angles: List of sets of three Euler angles for preferred orientations, optional.
    :param float angle_variation: Standard deviation for normal distribution of variations around preferred angles, optional.
    :param float preferred_weight: Weight of the preferred orientations in the range [0, 1], optional.
    :param int num_cores: Number of CPU cores to use for parallel processing, optional.
    :param bool use_gpu: Whether to use GPU for processing.
    :param list gpu_ids: List of GPU IDs to use for processing.
    :return numpy.ndarray list_of_tuples: Array of generated projections, and list of orientations as tuples of Euler angles (alpha, beta, gamma) in degrees.
    """
    print_and_log("", logging.DEBUG)
    projections = []

    if orientation_mode == 'random':
        orientations = [(np.random.uniform(0, 360), np.random.uniform(0, 180), np.random.uniform(0, 360)) for _ in range(num_projections)]
    elif orientation_mode == 'uniform':
        orientations = [(angle, np.random.uniform(0, 180), np.random.uniform(0, 360)) for angle in np.linspace(0, 360, num=num_projections, endpoint=False)]
    elif orientation_mode == 'preferred':
        if preferred_angles is None:
            preferred_angles = [['*', '*', '*']]
        else:
            preferred_angles = parse_preferred_angles(preferred_angles)
        num_preferred = int(num_projections * preferred_weight)
        num_random = num_projections - num_preferred
        orientations = generate_orientations(preferred_angles, angle_variation, num_preferred, num_random)

    np.random.shuffle(orientations)

    if use_gpu:
        # Trim the volume to the smallest possible cube containing non-zero voxels
        trimmed_volume = trim_volume(structure)

        # Determine the maximum batch size based on available GPU memory for each GPU
        slice_size = trimmed_volume.nbytes
        batch_sizes = {}

        for gpu_id in gpu_ids:
            utilization = get_gpu_utilization(gpu_id)
            batch_size = get_max_batch_size(slice_size, utilization['free_mem'])
            batch_sizes[gpu_id] = {'batch_size': batch_size, 'core_usage': utilization['core_usage']}

        # Distribute and process projections across available GPUs
        start = 0
        while start < num_projections:
            # Sort GPUs by core usage in ascending order
            sorted_gpus = sorted(batch_sizes.items(), key=lambda x: x[1]['core_usage'])
            for gpu_id, batch_info in sorted_gpus:
                end = start + batch_info['batch_size']
                projection_angles = orientations[start:end]
                if not projection_angles:
                    break

                # Process the batch of projections on the GPU
                with cp.cuda.Device(gpu_id):
                    trimmed_volume_gpu = cp.asarray(trimmed_volume)
                    for angle in projection_angles:
                        projection = generate_projection_gpu(angle, trimmed_volume_gpu, structure.shape[:2])
                        projections.append(projection)

                start = end
                if start >= num_projections:
                    break
    else:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(generate_projection, angle, structure) for angle in orientations]
            projections = [future.result() for future in futures]

    return np.array(projections), orientations

def generate_particle_locations(micrograph_image, image_size, num_small_images, half_small_image_width, 
                                border_distance, no_edge_particles, dist_type, non_random_dist_type, 
                                aggregation_amount, allow_overlap):
    """
    Generate random/non-random locations for particles within an image.

    :param numpy_array micrograph_image: Micrograph image (used only in the 'micrograph' distribution option).
    :param tuple image_size: The size of the image as a tuple (width, height).
    :param int num_small_images: The number of small images (particles) to generate coordinates for.
    :param int half_small_image_width: Half the width of a small image.
    :param int border_distance: The minimum distance between particles and the image border.
    :param bool no_edge_particles: Prevent particles from being placed up to the edge of the micrograph.
    :param str dist_type: Particle location generation distribution type - 'random' or 'non_random'.
    :param str non_random_dist_type: Type of non-random distribution when dist_type is 'non_random';
                                     ie. 'circular', 'inverse_circular', 'gaussian', or 'micrograph'.
    :param float aggregation_amount: Amount of particle aggregation.
    :param bool allow_overlap: Flag to allow overlapping particles.
    :return list_of_tuples: A list of particle locations as tuples (x, y).
    """
    print_and_log("", logging.DEBUG)
    width, height = image_size

    # If no_edge_particles is set, respect the user-defined --border value, 
    # otherwise use half of the particle box size as the default minimum distance from the edge
    border_distance = max(border_distance, half_small_image_width) if no_edge_particles else -1

    particle_locations = []

    def is_far_enough(new_particle_location, particle_locations, half_small_image_width, allow_overlap):
        """
        Check if a new particle location is far enough from existing particle locations.

        :param tuple new_particle_location: The new particle location as a tuple (x, y).
        :param list_of_tuples particle_locations: The existing particle locations.
        :param int half_small_image_width: Half the width of a small image.
        :param bool allow_overlap: Flag to allow overlapping particles.
        :return bool: True if the new particle location is far enough or if overlapping is allowed, False otherwise.
        """
        if allow_overlap:
            # Bypass the distance check if overlapping is allowed
            return True

        # Check distance to all existing particles
        for particle_location in particle_locations:
            distance = np.sqrt((new_particle_location[0] - particle_location[0])**2 + 
                               (new_particle_location[1] - particle_location[1])**2)
            if distance < half_small_image_width:
                return False  # Indicating overlap

        return True  # No overlap found

    max_attempts = 1000  # Maximum number of attempts to find an unoccupied point in the distribution

    if dist_type == 'random':
        attempts = 0  # Counter for attempts to find a valid position
        # Keep generating and appending particle locations until we have enough.
        while len(particle_locations) < num_small_images and attempts < max_attempts:
            x = np.random.randint(border_distance, width - border_distance)
            y = np.random.randint(border_distance, height - border_distance)
            new_particle_location = (x, y)
            
            if allow_overlap:
                # Directly add the new particle location if overlapping is allowed
                particle_locations.append(new_particle_location)
            else:
                # Check if the new particle location is far enough from existing ones
                if is_far_enough(new_particle_location, particle_locations, half_small_image_width, allow_overlap):
                    particle_locations.append(new_particle_location)
                    attempts = 0  # Reset attempts counter after successful addition
                else:
                    attempts += 1  # Increment attempts counter if addition is unsuccessful

    elif dist_type == 'non_random':
        # Handle non-random distributions (circular, inverse_circular, gaussian, micrograph)
        if non_random_dist_type == 'circular':
            # Make a circular cluster of particles
            cluster_center = (width // 2, height // 2)
            attempts = 0  # Counter for attempts to find a valid position

            # Keep generating and appending particle locations until we have enough.
            while len(particle_locations) < num_small_images and attempts < max_attempts:
                # Generate random angle and radius for polar coordinates.
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, min(cluster_center[0], height - cluster_center[1]) - half_small_image_width)
                # Convert polar to Cartesian coordinates.
                x = int(cluster_center[0] + radius * np.cos(angle))
                y = int(cluster_center[1] + radius * np.sin(angle))
                new_particle_location = (x, y)
                
                if allow_overlap:
                    particle_locations.append(new_particle_location)
                else:
                    if is_far_enough(new_particle_location, particle_locations, half_small_image_width, allow_overlap):
                        particle_locations.append(new_particle_location)
                        attempts = 0
                    else:
                        attempts += 1

        elif non_random_dist_type == 'inverse_circular':
            # Parameters for the exclusion zone
            # Randomly determine the center within the image, away from the edges
            exclusion_center_x = np.random.randint(border_distance + half_small_image_width, width - border_distance - half_small_image_width)
            exclusion_center_y = np.random.randint(border_distance + half_small_image_width, height - border_distance - half_small_image_width)
            exclusion_radius = np.random.randint(half_small_image_width, min(width // 2, height // 2))

            attempts = 0
            while len(particle_locations) < num_small_images and attempts < max_attempts:
                x = np.random.randint(border_distance, width - border_distance)
                y = np.random.randint(border_distance, height - border_distance)
                new_particle_location = (x, y)
                # Check if the location is outside the exclusion zone
                if np.sqrt((x - exclusion_center_x) ** 2 + (y - exclusion_center_y) ** 2) > exclusion_radius:
                    if allow_overlap:
                        particle_locations.append(new_particle_location)
                    else:
                        if is_far_enough(new_particle_location, particle_locations, half_small_image_width, allow_overlap):
                            particle_locations.append(new_particle_location)
                            attempts = 0
                        else:
                            attempts += 1
                else:
                    attempts += 1

        elif non_random_dist_type == 'gaussian':
            num_gaussians = np.random.randint(1, 6)
            gaussians = []
            # For each Gaussian distribution:
            for _ in range(num_gaussians):
                # Randomly determine its center and standard deviation.
                center = np.array([np.random.uniform(border_distance, width - border_distance),
                                np.random.uniform(border_distance, height - border_distance)])
                stddev = np.random.uniform(half_small_image_width, min(center[0] - border_distance,
                                                                    width - center[0] - border_distance,
                                                                    center[1] - border_distance,
                                                                    height - center[1] - border_distance))
                gaussians.append((center, stddev))

            attempts = 0
            while len(particle_locations) < num_small_images and attempts < max_attempts:
                chosen_gaussian = np.random.choice(num_gaussians)
                center, stddev = gaussians[chosen_gaussian]
                x = int(np.random.normal(center[0], stddev))
                y = int(np.random.normal(center[1], stddev))
                new_particle_location = (x, y)
                
                if allow_overlap:
                    particle_locations.append(new_particle_location)
                else:
                    if border_distance <= x <= width - border_distance and border_distance <= y <= height - border_distance and is_far_enough(new_particle_location, particle_locations, half_small_image_width, allow_overlap):
                        particle_locations.append(new_particle_location)
                        attempts = 0
                    else:
                        attempts += 1

        elif non_random_dist_type == 'micrograph':
            # Apply a Gaussian filter to the micrograph to obtain large-scale features
            sigma = min(width, height) / 1  # Keep only low-resolution features. Something needs to be in the denominator otherwise the smoothing function breaks
            itk_image = sitk.GetImageFromArray(micrograph_image)
            filtered_micrograph_itk = sitk.SmoothingRecursiveGaussian(itk_image, sigma=sigma)
            filtered_micrograph = sitk.GetArrayFromImage(filtered_micrograph_itk)

            # Invert the filtered image to assign higher probability to lower pixel values
            inverted_micrograph = np.max(filtered_micrograph) - filtered_micrograph

            # Normalize the inverted image to get probabilities
            prob_map = inverted_micrograph / inverted_micrograph.sum()

            # Flatten the probability map and generate indices
            flat_prob_map = prob_map.ravel()

            # Generate a large batch of random choices before the loop
            batch_size = num_small_images * 10
            random_indices = np.random.choice(flat_prob_map.size, size=batch_size, p=flat_prob_map)

            # Initialize clump centers
            num_clumps = max(1, int(num_small_images * np.random.uniform(0.1, 0.5)))
            clump_centers = [(np.random.randint(border_distance, width - border_distance),
                              np.random.randint(border_distance, height - border_distance)) for _ in range(num_clumps)]

            # Initialize the attempt counter for 'micrograph' distribution
            attempts = 0  # Reset attempts counter
            index_counter = 0  # Counter to iterate through the batch of random choices
            while len(particle_locations) < num_small_images and index_counter < batch_size and attempts < max_attempts:
                chosen_index = random_indices[index_counter]
                index_counter += 1
                y, x = divmod(chosen_index, width)

                if aggregation_amount > 0 and particle_locations:
                    aggregation_factor = aggregation_amount / 11.0
                    clump_center = clump_centers[np.random.choice(len(clump_centers))]
                    shift_x = int((clump_center[0] - x) * aggregation_factor)
                    shift_y = int((clump_center[1] - y) * aggregation_factor)
                    # To make it so clumps aren't universal attractors, only update the particle location if
                    # aggregation_factor is less than a random number (ie. use this as a probability of changing)
                    if random.random() <= aggregation_factor:
                        new_particle_location = (x + shift_x, y + shift_y)
                    else:
                        new_particle_location = (x, y)
                else:
                    new_particle_location = (x, y)

                # Check if the new location is within borders and far enough from other particles
                if border_distance <= x <= width - border_distance and border_distance <= y <= height - border_distance and is_far_enough(new_particle_location, particle_locations, half_small_image_width, allow_overlap):
                    particle_locations.append(new_particle_location)
                    attempts = 0  # Reset attempts counter after successful addition
                else:
                    attempts += 1  # Increment attempts counter if addition is unsuccessful

    if non_random_dist_type == 'micrograph':
        # This will make a non-linear gradient from the probability map (ice thickness gradient) that will scale the ice thickness +-20% from the given value
        # 1 - prob_map because particles should be scaled to be darker in thinner areas and lighter in thinner areas
        inverse_normalized_prob_map = 0.8 + 0.4 * (1 - (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min()))
        return particle_locations, inverse_normalized_prob_map
    else:
        return particle_locations, None

def estimate_noise_parameters(image):
    """
    Estimate Poisson (shot noise) and Gaussian (readout and electronic noise) parameters from a 2D image.

    :param numpy_array image: 2D numpy array representing the image.
    :return float, float: A tuple containing the estimated Poisson variance (mean) and Gaussian variance.
    """
    print_and_log("", logging.DEBUG)
    # Calculate mean and variance across the image
    mean = np.mean(image)
    variance = np.var(image)

    # Poisson noise component is approximated by the mean
    # Gaussian noise component is the excess variance over the Poisson component
    gaussian_variance = variance - mean

    return mean, gaussian_variance

def process_slices_gpu(args):
    """
    Process slices of the particle stack by adding Poisson noise and optionally dose damaging using GPU.

    :param args: A tuple containing the following parameters:
                 - slice numpy_array: A 3D numpy array representing a slice of the particle stack.
                 - num_frames int: Number of frames to simulate for each particle image.
                 - float dose_a: Custom value for the 'a' variable in equation (3) of Grant & Grigorieff, 2015.
                 - float dose_b: Custom value for the 'b' variable in equation (3) of Grant & Grigorieff, 2015.
                 - float dose_c: Custom value for the 'c' variable in equation (3) of Grant & Grigorieff, 2015.
                 - float apix: Pixel size (in Angstroms) of the ice images.
                 - scaling_factor float: Factor by which to scale the particle images before adding noise.
    :return numpy_array: A 3D numpy array representing the processed slice of the particle stack with added noise.
    """
    print_and_log("", logging.DEBUG)
    # Unpack the arguments
    slice, num_frames, scaling_factor, dose_a, dose_b, dose_c, apix = args

    # Transfer slice to GPU
    slice_gpu = cp.asarray(slice, dtype=cp.float32)
    noisy_slice_gpu = cp.zeros_like(slice_gpu)

    # Iterate over each particle in the slice
    for i in range(slice_gpu.shape[0]):
        # Get the i-th particle and scale it by the scaling factor
        particle = slice_gpu[i, :, :] * scaling_factor

        # Create a mask for non-zero values in the particle
        mask = particle > 0

        # For each frame, simulate the noise and accumulate the result
        for j in range(num_frames):
            j += 1  # Simulates the accumulated dose per 1 e/A^2 per simulated frame

            # Initialize a frame with zeros
            noisy_frame_gpu = cp.zeros_like(particle)

            # Add Poisson noise to the non-zero values in the particle, modulated by the original pixel values; it represents shot noise.
            noisy_frame_gpu[mask] = cp.random.poisson(particle[mask])

            # Dose damage frames if requested
            if not (dose_a == dose_b == dose_c == 0):  # All zero is equivalent to the user specifying None for dose_damage
                if j != dose_c:
                    lowpass = float(cp.real(cp.complex64(dose_a/(j - dose_c))**(1/dose_b)))  # Equation (3) from Grant & Grigorieff, 2015. Assumes 1 e/A^2 per simulated frame.
                else:  # No divide by zero
                    lowpass = float(cp.real(cp.complex64(dose_a/(j - dose_c + 0.001))**(1/dose_b)))  # Slight perturbation of the equation above

                # Apply low-pass filter on the GPU
                noisy_frame_gpu = lowPassFilter_gpu(noisy_frame_gpu, apix=apix, radius=lowpass)

            # Accumulate the noisy frame into the noisy slice
            noisy_slice_gpu[i, :, :] += noisy_frame_gpu

    return cp.asnumpy(noisy_slice_gpu)

def process_slice(args):
    """
    Process a slice of the particle stack by adding Poisson noise to simulate electron counts from proteins.
    Optionally applies dose damage to the frames.

    :param args: A tuple containing the following parameters:
                 - slice numpy_array: A 3D numpy array representing a slice of the particle stack.
                 - num_frames int: Number of frames to simulate for each particle image.
                 - float dose_a: Custom value for the 'a' variable in equation (3) of Grant & Grigorieff, 2015.
                 - float dose_b: Custom value for the 'b' variable in equation (3) of Grant & Grigorieff, 2015.
                 - float dose_c: Custom value for the 'c' variable in equation (3) of Grant & Grigorieff, 2015.
                 - float apix: Pixel size (in Angstroms) of the ice images.
                 - scaling_factor float: Factor by which to scale the particle images before adding noise.
    :return numpy_array: A 3D numpy array representing the processed slice of the particle stack with added noise.
    """
    print_and_log("", logging.DEBUG)
    # Unpack the arguments
    slice, num_frames, scaling_factor, dose_a, dose_b, dose_c, apix = args

    # Create an empty array to store the noisy slice of the particle stack
    noisy_slice = np.zeros_like(slice, dtype=np.float32)

    # Iterate over each particle in the slice
    for i in range(slice.shape[0]):
        # Get the i-th particle and scale it by the scaling factor
        particle = slice[i, :, :] * scaling_factor

        # Create a mask for non-zero values in the particle
        mask = particle > 0

        # For each frame, simulate the noise and accumulate the result
        for j in range(num_frames):
            j += 1  # Simulates the accumulated dose per 1 e/A^2 per simulated frame
            # Initialize a frame with zeros
            noisy_frame = np.zeros_like(particle, dtype=np.float32)

            # Add Poisson noise to the non-zero values in the particle, modulated by the original pixel values; it represents shot noise.
            noisy_frame[mask] = np.random.poisson(particle[mask])

            # Dose damage frames if requested
            if not (dose_a == dose_b == dose_c == 0):  # All zero is equivalent to the user specifying None for dose_damage
                if j != dose_c:
                    lowpass = float(np.real(complex(dose_a/(j - dose_c))**(1/dose_b)))  # Equation (3) from Grant & Grigorieff, 2015. Assumes 1 e/A^2 per simulated frame.
                else:  #No divide by zero
                    lowpass = float(np.real(complex(dose_a/(j - dose_c + 0.001))**(1/dose_b)))  # Slight perturbation of the equation above
                noisy_frame = lowPassFilter(noisy_frame, apix=apix, radius=lowpass)

            # Accumulate the noisy frame into the noisy slice
            noisy_slice[i, :, :] += noisy_frame

    return noisy_slice

def add_poisson_noise_gpu(particle_stack, num_frames, dose_a, dose_b, dose_c, apix, gpu_ids, scaling_factor=1.0):
    """
    Add Poisson noise to a stack of particle images using GPU.

    This function simulates the acquisition of `num_frames` frames for each particle image
    in the input stack, adds Poisson noise to each frame, and then sums up the frames to
    obtain the final noisy particle image. The function applies both noises only to the
    non-zero values in each particle image, preserving the background.

    :param numpy_array particle_stack: 3D numpy array representing a stack of 2D particle images.
    :param int num_frames: Number of frames to simulate for each particle image.
    :param float dose_a: Custom value for the 'a' variable in equation (3) of Grant & Grigorieff, 2015.
    :param float dose_b: Custom value for the 'b' variable in equation (3) of Grant & Grigorieff, 2015.
    :param float dose_c: Custom value for the 'c' variable in equation (3) of Grant & Grigorieff, 2015.
    :param float apix: Pixel size (in Angstroms) of the ice images.
    :param float scaling_factor: Factor by which to scale the particle images before adding noise.
    :param list gpu_ids: List of GPU IDs to use for processing.
    :return numpy_array: 3D numpy array representing the stack of noisy particle images.
    """
    print_and_log("", logging.DEBUG)

    # Determine the maximum batch size based on available GPU memory for each GPU
    slice_size = particle_stack[0].nbytes
    batch_sizes = {}

    for gpu_id in gpu_ids:
        utilization = get_gpu_utilization(gpu_id)
        batch_size = get_max_batch_size(slice_size, utilization['free_mem'])
        batch_sizes[gpu_id] = {'batch_size': batch_size, 'core_usage': utilization['core_usage']}

    num_slices = particle_stack.shape[0]
    noisy_particle_stack = []

    # Distribute and process slices across available GPUs
    start = 0
    while start < num_slices:
        # Sort GPUs by core usage in ascending order
        sorted_gpus = sorted(batch_sizes.items(), key=lambda x: x[1]['core_usage'])
        for gpu_id, batch_info in sorted_gpus:
            end = start + batch_info['batch_size']
            slice_chunk = particle_stack[start:end]
            if slice_chunk.shape[0] == 0:
                break

            # Process the chunk on the GPU
            with cp.cuda.Device(gpu_id):
                noisy_chunk = process_slices_gpu((slice_chunk, num_frames, scaling_factor, dose_a, dose_b, dose_c, apix))
            noisy_particle_stack.append(noisy_chunk)
            start = end
            if start >= num_slices:
                break

    # Concatenate the processed chunks back into a single stack
    noisy_particle_stack = np.concatenate(noisy_particle_stack, axis=0)
    return noisy_particle_stack

def add_poisson_noise(particle_stack, num_frames, dose_a, dose_b, dose_c, apix, num_cores, scaling_factor=1.0):
    """
    Add Poisson noise to a stack of particle images.

    This function simulates the acquisition of `num_frames` frames for each particle image
    in the input stack, adds Poisson noise to each frame, and then sums up the frames to
    obtain the final noisy particle image. The function applies both noises only to the
    non-zero values in each particle image, preserving the background.

    :param numpy_array particle_stack: 3D numpy array representing a stack of 2D particle images.
    :param int num_frames: Number of frames to simulate for each particle image.
    :param float dose_a: Custom value for the 'a' variable in equation (3) of Grant & Grigorieff, 2015.
    :param float dose_b: Custom value for the 'b' variable in equation (3) of Grant & Grigorieff, 2015.
    :param float dose_c: Custom value for the 'c' variable in equation (3) of Grant & Grigorieff, 2015.
    :param float apix: Pixel size (in Angstroms) of the ice images.
    :param int num_cores: Number of CPU cores to parallelize slices across.
    :param float scaling_factor: Factor by which to scale the particle images before adding noise.
    :return numpy_array: 3D numpy array representing the stack of noisy particle images.
    """
    print_and_log("", logging.DEBUG)
    # Split the particle stack into slices
    slices = np.array_split(particle_stack, num_cores)

    # Prepare the arguments for each slice
    args = [(s, num_frames, scaling_factor, dose_a, dose_b, dose_c, apix) for s in slices]

    # Create a pool of worker processes
    with Pool(num_cores) as pool:
        # Process each slice in parallel
        noisy_slices = pool.map(process_slice, args)

    # Concatenate the processed slices back into a single stack
    noisy_particle_stack = np.concatenate(noisy_slices, axis=0)

    return noisy_particle_stack

def apply_ctf_with_eman2(particle, defocus, params):
    """
    Apply CTF to a particle with EMAN2.

    :param tuple params: Tuple containing (ampcont, bfactor, apix, cs, voltage)
    :param numpy.ndarray particle: The particle to be processed
    :param float defocus: Defocus value for CTF simulation
    :return numpy.ndarray: Processed particle as a float32 NumPy array
    """
    print_and_log("", logging.DEBUG)
    ampcont, bfactor, apix, cs, voltage = params
    input_particle = EMNumPy.numpy2em(particle)

    # Apply CTF simulation, multiply by -1, and normalize
    ctf_params = {"ampcont": ampcont, "bfactor": bfactor, "apix": apix, "cs": cs, "defocus": defocus, "voltage": voltage}
    input_particle.process_inplace("math.simulatectf", ctf_params)
    input_particle.mult(-1)
    input_particle.process_inplace("normalize.edgemean")

    # Convert the processed EMData object back to a NumPy array and cast to float32
    particle_CTF = EMNumPy.em2numpy(input_particle).astype(np.float32)
    return particle_CTF

def apply_ctfs_with_eman2(particles, defocuses, ampcont, bfactor, apix, cs, voltage, num_workers):
    """
    Apply CTF to a stack of particles using EMAN2 (parallelized).

    :param numpy.ndarray particles: Stack of particles to be processed
    :param list defocuses: List of defocus values, one for each particle in the stack
    :param float ampcont: Amplitude contrast for CTF simulation
    :param float bfactor: B-factor for CTF simulation
    :param float apix: Pixel size for CTF simulation
    :param float cs: Spherical aberration for CTF simulation
    :param float voltage: Voltage for CTF simulation
    :param int num_workers: Number of CPU cores to use for parallel processing
    :return numpy.ndarray: Stack of processed particles as a float32 NumPy array
    """
    print_and_log("", logging.DEBUG)
    params = (ampcont, bfactor, apix, cs, voltage)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(apply_ctf_with_eman2, particles, defocuses, [params]*len(particles)))

    particles_CTF = np.array(results, dtype=np.float32)
    return particles_CTF

def filter_out_overlapping_particles(particle_locations, half_small_image_width):
    """
    Filter out overlapping particles based on the center-to-center distance.

    :param list_of_tuples particle_locations: List of (x, y) coordinates of particle locations.
    :param int half_small_image_width: Half the width of a small image.
    :return list_of_tuples: List of (x, y) coordinates of non-overlapping particle locations.
    """
    print_and_log("", logging.DEBUG)

    # Use scipy.spatial.KDTree for efficient nearest-neighbor search
    from scipy.spatial import KDTree

    # Build a KDTree for the particle locations
    tree = KDTree(particle_locations)

    # Define the minimum distance required to avoid overlap (0.92 is a fudge factor)
    min_distance = 0.92 * half_small_image_width

    # Find all pairs of particles that are closer than min_distance
    overlapping_particles = set()
    for i, location in enumerate(particle_locations):
        indices = tree.query_ball_point(location, min_distance)
        # If more than one particle is found within the min_distance, mark them as overlapping
        if len(indices) > 1:
            overlapping_particles.update(indices)

    # Filter out the overlapping particles
    filtered_locations = [loc for i, loc in enumerate(particle_locations) if i not in overlapping_particles]

    return filtered_locations

def create_collage(large_image, small_images, particle_locations, gaussian_variance):
    """
    Create a collage of small images on a blank canvas of the same size as the large image.
    Add Gaussian noise based on the micrograph that the particle collage will be projected onto.
    Note: Gaussian noise is characteristic of the microscope & camera and is independent of the particles' signal.
    Particles that would extend past the edge of the large image are trimmed before being added.

    :param numpy.ndarray large_image: Shape of the large image.
    :param list_of_numpy.ndarray small_images: List of small images to place on the canvas.
    :param list_of_tuples particle_locations: Coordinates where each small image should be placed.
    :param float gaussian_variance: Standard deviation of the Gaussian noise, as measured previously from the ice image.
    :return numpy.ndarray: Collage composed of small images.
    """
    print_and_log("", logging.DEBUG)
    collage = np.zeros(large_image.shape, dtype=large_image.dtype)

    for i, small_image in enumerate(small_images):
        x, y = particle_locations[i]
        x_start = x - small_image.shape[1] // 2
        y_start = y - small_image.shape[0] // 2

        x_end = x_start + small_image.shape[1]
        y_end = y_start + small_image.shape[0]

        # Calculate the region of the small image that fits within the large image
        x_start_trim = max(0, -x_start)
        y_start_trim = max(0, -y_start)
        x_end_trim = min(small_image.shape[1], large_image.shape[1] - x_start)
        y_end_trim = min(small_image.shape[0], large_image.shape[0] - y_start)

        # Adjust start and end coordinates to ensure they fall within the large image
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(large_image.shape[1], x_end)
        y_end = min(large_image.shape[0], y_end)

        # Ensure the sizes match by explicitly setting the shape dimensions
        trim_height = min(y_end - y_start, y_end_trim - y_start_trim)
        trim_width = min(x_end - x_start, x_end_trim - x_start_trim)

        # Adjust dimensions to match
        y_end = y_start + trim_height
        x_end = x_start + trim_width
        y_end_trim = y_start_trim + trim_height
        x_end_trim = x_start_trim + trim_width

        # Ensure dimensions match before addition
        collage_region = collage[y_start:y_end, x_start:x_end]
        small_image_region = small_image[y_start_trim:y_end_trim, x_start_trim:x_end_trim]

        if collage_region.shape == small_image_region.shape:
            collage[y_start:y_end, x_start:x_end] += small_image_region
        else:
            min_height = min(collage_region.shape[0], small_image_region.shape[0])
            min_width = min(collage_region.shape[1], small_image_region.shape[1])
            collage[y_start:y_start + min_height, x_start:x_start + min_width] += small_image_region[:min_height, :min_width]

    # Apply Gaussian noise across the entire collage to simulate the camera noise
    gaussian_noise = np.random.normal(loc=0, scale=np.sqrt(gaussian_variance), size=collage.shape)
    collage += gaussian_noise

    return collage

def blend_images(input_options, particle_and_micrograph_generation_options, simulation_options, 
                 junk_labels_options, output_options, context, defocus):
    """
    Blend small images (particles) into a large image (micrograph).
    Also makes coordinate files.

    :param dict input_options: Dictionary of input options, including large_image, small_images, etc.
    :param dict particle_and_micrograph_generation_options: Dictionary of particle and micrograph generation options.
    :param dict simulation_options: Dictionary of simulation options.
    :param dict junk_labels_options: Dictionary of options for junk labels.
    :param dict output_options: Dictionary of output options.
    :param str context: Context string for print statements (structure name and micrograph number).
    :param float defocus: The defocus value to add to the STAR file.
    :return tuple: The blended large image, and the filtered particle locations.
    """
    print_and_log("", logging.DEBUG)
    # Extract input options
    large_image = input_options['large_image']
    small_images = input_options['small_images']
    particle_locations = input_options['particle_locations']
    orientations = input_options['orientations']
    structure_name = input_options['structure_name']
    output_path = output_options['output_path']

    # Extract junk labels options
    no_junk_filter = junk_labels_options['no_junk_filter']
    flip_x = junk_labels_options['flip_x']
    flip_y = junk_labels_options['flip_y']
    json_scale = junk_labels_options['json_scale']
    polygon_expansion_distance = junk_labels_options['polygon_expansion_distance']

    # Junk Filtering
    json_file_path = os.path.splitext(input_options['large_image_path'])[0] + ".json"
    if not no_junk_filter:
        if os.path.exists(json_file_path):
            polygons = read_polygons_from_json(json_file_path, polygon_expansion_distance, flip_x, flip_y, expand=True)
            # Remove particle locations from inside polygons (junk in micrographs) when writing coordinate files
            filtered_particle_locations = filter_coordinates_outside_polygons(particle_locations, json_scale, polygons)
            num_particles_removed = len(particle_locations) - len(filtered_particle_locations)
            print_and_log(f"{context} {num_particles_removed} particle{'' if num_particles_removed == 1 else 's'} removed from coordinate file(s) based on the JSON file.")
        else:
            print_and_log(f"{context} No JSON file found for bad micrograph areas: {json_file_path}", logging.WARNING)
            filtered_particle_locations = particle_locations
    else:
        print_and_log(f"{context} Skipping junk filtering (i.e., not using JSON file)")
        filtered_particle_locations = particle_locations

    # Edge Particle Filtering
    if not particle_and_micrograph_generation_options['save_edge_coordinates']:
        remaining_particle_locations = filtered_particle_locations[:]
        for x, y in filtered_particle_locations:
            reduced_sidelength = int(np.ceil(input_options['half_small_image_width'] * 100 / (100 + particle_and_micrograph_generation_options['scale_percent'])))
            left_edge = x - reduced_sidelength
            right_edge = x + reduced_sidelength
            top_edge = y - reduced_sidelength
            bottom_edge = y + reduced_sidelength

            # Determine if the particle is too close to any edge of the large image
            if (left_edge < particle_and_micrograph_generation_options['border_distance'] or 
                right_edge > large_image.shape[1] - particle_and_micrograph_generation_options['border_distance'] or
                top_edge < particle_and_micrograph_generation_options['border_distance'] or 
                bottom_edge > large_image.shape[0] - particle_and_micrograph_generation_options['border_distance']):
                remaining_particle_locations.remove((x, y))

        num_particles_removed = len(filtered_particle_locations) - len(remaining_particle_locations)
        if num_particles_removed > 0:
            print_and_log(f"{context} {num_particles_removed} particle{'' if num_particles_removed == 1 else 's'} removed from coordinate file(s) due to being too close to the edge.")
        else:
            print_and_log(f"{context} 0 particles removed from coordinate file(s) due to being too close to the edge.")
        filtered_particle_locations = remaining_particle_locations

    # Overlapping Particle Filtering - for Coordinate Files only
    if not particle_and_micrograph_generation_options['save_overlapping_coords']:
        filtered_particle_locations = filter_out_overlapping_particles(filtered_particle_locations, input_options['half_small_image_width'])
        num_particles_removed = len(particle_locations) - len(filtered_particle_locations)
        if num_particles_removed > 0:
            print_and_log(f"{context} {num_particles_removed} overlapping particle{'' if num_particles_removed == 1 else 's'} removed from coordinate file(s).")
        else:
            print_and_log(f"{context} 0 particles removed from coordinate file(s) due to overlapping.")

    # Ensure small_images and particle_locations are the same length
    if len(small_images) > len(particle_locations):
        small_images = small_images[:len(particle_locations)]

    # Normalize the input micrograph to itself
    large_image[:, :] = (large_image[:, :] - large_image[:, :].mean()) / large_image[:, :].std()

    # Create the collage of particles on the micrograph
    collage = create_collage(large_image, small_images, particle_locations, 
                             particle_and_micrograph_generation_options['gaussian_variance'])

    # If a probability map is provided, adjust the collage based on local ice thickness
    if 'prob_map' in input_options and input_options['prob_map'] is not None:
        collage *= simulation_options['scale'] * input_options['prob_map']
    else:
        collage *= simulation_options['scale']

    # Blend the collage with the large image
    blended_image = large_image + collage

    # Normalize the resulting micrograph to itself
    blended_image = (blended_image - blended_image.mean()) / blended_image.std()

    # Combine filtered_particle_locations with orientations for easier passing
    filtered_particle_locations_with_orientations = [(loc, ori) for loc, ori in zip(filtered_particle_locations, orientations)]

    # Save particle coordinates to coordinate files
    save_particle_coordinates(structure_name, filtered_particle_locations_with_orientations, output_path, 
                              output_options['imod_coordinate_file'], output_options['coord_coordinate_file'], defocus)

    return blended_image, filtered_particle_locations_with_orientations

def add_images(input_options, particle_and_micrograph_generation_options, simulation_options, 
               junk_labels_options, output_options, context, defocus):
    """
    Add small images or particles to a large image and save the resulting micrograph.

    :param dict input_options: Dictionary of input options, including large_image, small_images, etc.
    :param dict particle_and_micrograph_generation_options: Dictionary of particle and micrograph generation options.
    :param dict simulation_options: Dictionary of simulation options.
    :param dict junk_labels_options: Dictionary of options for junk labels.
    :param dict output_options: Dictionary of output options.
    :param str context: Context string for print statements (structure name and micrograph number).
    :param float defocus: The defocus value to add to the STAR file.
    :return int, int: The number of particles added to the micrograph, and the number of particles saved to coordinate file(s).

    This function writes a .mrc/.png/.jpeg file to the disk.
    """
    print_and_log("", logging.DEBUG)

    # Extract input options
    large_image_path = input_options['large_image_path']
    large_image = input_options['large_image']
    small_images = input_options['small_images']
    pixelsize = input_options['pixelsize']
    structure_name = input_options['structure_name']
    orientations = input_options['orientations']
    output_path = output_options['output_path']

    # Calculate half the width of a small image
    half_small_image_width = int(small_images.shape[1] / 2)

    # Extract particle and micrograph generation options
    scale_percent = particle_and_micrograph_generation_options['scale_percent']
    dist_type = particle_and_micrograph_generation_options['dist_type']
    non_random_dist_type = particle_and_micrograph_generation_options['non_random_dist_type']
    border_distance = particle_and_micrograph_generation_options['border_distance']
    no_edge_particles = particle_and_micrograph_generation_options['no_edge_particles']
    save_edge_coordinates = particle_and_micrograph_generation_options['save_edge_coordinates']
    gaussian_variance = particle_and_micrograph_generation_options['gaussian_variance']
    aggregation_amount = particle_and_micrograph_generation_options['aggregation_amount']
    allow_overlap = particle_and_micrograph_generation_options['allow_overlap']

    # Generate particle locations using generate_particle_locations function
    particle_locations, prob_map = generate_particle_locations(large_image, np.flip(large_image.shape), 
                                                               len(small_images), half_small_image_width, 
                                                               border_distance, no_edge_particles, dist_type, 
                                                               non_random_dist_type, aggregation_amount, allow_overlap)

    # Add changed variables to input_options
    input_options['prob_map'] = prob_map
    input_options['particle_locations'] = particle_locations
    input_options['half_small_image_width'] = half_small_image_width

    # Proceed with blending images
    if len(particle_locations) != len(small_images):
        print_and_log(f"{context} Only {len(particle_locations)} could fit into the image. Adding those to the micrograph now...")
    result_image, filtered_particle_locations = blend_images(input_options, particle_and_micrograph_generation_options,
                                                             simulation_options, junk_labels_options, output_options, context, defocus)

    # Save the resulting micrograph in specified formats
    if output_options['save_as_mrc']:
        print_and_log(f"\n{context} Writing synthetic micrograph: {output_path}.mrc...")
        writemrc(output_path + '.mrc', (result_image - np.mean(result_image)) / np.std(result_image), pixelsize)  # Write normalized mrc (mean = 0, std = 1)
    if output_options['save_as_png']:
        # Needs to be scaled from 0 to 255 and flipped
        result_image -= result_image.min()
        result_image = result_image / result_image.max() * 255.0
        print_and_log(f"\n{context} Writing synthetic micrograph: {output_path}.png...")
        cv2.imwrite(output_path + '.png', np.flip(result_image, axis=0))
    if output_options['save_as_jpeg']:
        # Needs to be scaled from 0 to 255 and flipped
        result_image -= result_image.min()
        result_image = result_image / result_image.max() * 255.0
        print_and_log(f"\n{context} Writing synthetic micrograph: {output_path}.jpeg...")
        cv2.imwrite(output_path + '.jpeg', np.flip(result_image, axis=0), [cv2.IMWRITE_JPEG_QUALITY, output_options['jpeg_quality']])

    return len(particle_locations), len(filtered_particle_locations)

def crop_particles_gpu(micrograph_path, particle_rows, particles_dir, box_size, pixelsize, max_crop_particles, gpu_id):
    """
    Crops particles from multiple micrographs using GPU in batches.

    :param list micrograph_paths: List of paths to the micrographs.
    :param list particle_rows_list: List of DataFrame rows of particles to be cropped from each micrograph.
    :param str particles_dir: Directory to save cropped particles.
    :param int box_size: The box size in pixels for the cropped particles.
    :param float pixelsize: Pixel size of the micrographs.
    :param int max_crop_particles: The maximum number of particles to crop from each micrograph.
    :param int gpu_id: ID of the GPU to use for processing.
    :return int: Total number of particles cropped from the micrographs.
    """
    print_and_log("", logging.DEBUG)
    cropped_count = 0
    with cp.cuda.Device(gpu_id):
        with mrcfile.open(micrograph_path, permissive=True) as mrc:
            micrograph_data = cp.asarray(mrc.data)
            for _, row in particle_rows.iterrows():
                if max_crop_particles and cropped_count >= max_crop_particles:
                    break
                x, y = int(row['coord_x']), int(row['coord_y'])
                half_box_size = box_size // 2

                if x - half_box_size < 0 or y - half_box_size < 0 or x + half_box_size > micrograph_data.shape[1] or y + half_box_size > micrograph_data.shape[0]:
                    continue

                cropped_particle = micrograph_data[y-half_box_size:y+half_box_size, x-half_box_size:x+half_box_size]
                particle_path = os.path.join(particles_dir, f"particle_{row['particle_counter']:010d}.mrc")
                writemrc(particle_path, cp.asnumpy(cropped_particle).astype(np.float32), pixelsize)
                cropped_count += 1

    return cropped_count

def crop_particles_gpu_batch(micrograph_paths, particle_rows_list, particles_dir, box_size, pixelsize, max_crop_particles, gpu_id):
    """
    Crops particles from micrographs based on coordinates specified in a micrograph STAR file
    and saves them with a specified box size. This function operates in parallel, using multiple GPUs
    or CPU cores to process different micrographs concurrently.

    :param str structure_dir: The directory containing the structure's micrographs and STAR file.
    :param int box_size: The box size in pixels for the cropped particles.
    :param float pixelsize: Pixel size of the micrographs.
    :param int num_cpus: Number of CPU cores for parallel processing if GPUs are not used.
    :param int max_crop_particles: The maximum number of particles to crop from the micrographs.
    :param bool use_gpu: Whether to use GPU for processing.
    :param list gpu_ids: List of GPU IDs to use for processing.
    :return int: Total number of particles cropped from the micrographs.
    """
    print_and_log("", logging.DEBUG)
    total_cropped = 0
    with cp.cuda.Device(gpu_id):
        for micrograph_path, particle_rows in zip(micrograph_paths, particle_rows_list):
            with mrcfile.open(micrograph_path, permissive=True) as mrc:
                micrograph_data = cp.asarray(mrc.data)
                cropped_count = 0
                for _, row in particle_rows.iterrows():
                    if max_crop_particles and cropped_count >= max_crop_particles:
                        break
                    x, y = int(row['coord_x']), int(row['coord_y'])
                    half_box_size = box_size // 2

                    if x - half_box_size < 0 or y - half_box_size < 0 or x + half_box_size > micrograph_data.shape[1] or y + half_box_size > micrograph_data.shape[0]:
                        continue

                    cropped_particle = micrograph_data[y-half_box_size:y+half_box_size, x-half_box_size:x+half_box_size]
                    particle_path = os.path.join(particles_dir, f"particle_{row['particle_counter']:010d}.mrc")
                    writemrc(particle_path, cp.asnumpy(cropped_particle).astype(np.float32), pixelsize)
                    cropped_count += 1
                total_cropped += cropped_count

    return total_cropped

def crop_particles(micrograph_path, particle_rows, particles_dir, box_size, pixelsize, max_crop_particles):
    """
    Crops particles from a single micrograph.

    :param str micrograph_path: Path to the micrograph.
    :param DataFrame particle_rows: DataFrame rows of particles to be cropped from the micrograph.
    :param str particles_dir: Directory to save cropped particles.
    :param int box_size: The box size in pixels for the cropped particles.
    :param float pixelsize: Pixel size of the micrograph.
    :param int max_crop_particles: The maximum number of particles to crop from the micrograph.
    :return int: Total number of particles cropped from the micrograph.

    This function writes .mrc files to the disk.
    """
    print_and_log("", logging.DEBUG)
    cropped_count = 0
    with mrcfile.open(micrograph_path, permissive=True) as mrc:
        for _, row in particle_rows.iterrows():
            if max_crop_particles and cropped_count >= max_crop_particles:
                break
            x, y = int(row['coord_x']), int(row['coord_y'])
            half_box_size = box_size // 2

            if x - half_box_size < 0 or y - half_box_size < 0 or x + half_box_size > mrc.data.shape[1] or y + half_box_size > mrc.data.shape[0]:
                continue

            cropped_particle = mrc.data[y-half_box_size:y+half_box_size, x-half_box_size:x+half_box_size]
            particle_path = os.path.join(particles_dir, f"particle_{row['particle_counter']:010d}.mrc")
            writemrc(particle_path, cropped_particle.astype(np.float32), pixelsize)
            cropped_count += 1

    return cropped_count

def crop_particles_from_micrographs(structure_dir, box_size, pixelsize, max_crop_particles, num_cpus, use_gpu, gpu_ids):
    """
    Crops particles from micrographs based on coordinates specified in a micrograph STAR file
    and saves them with a specified box size. This function operates in parallel, using multiple GPUs
    or multiple CPU cores to process different micrographs concurrently.

    :param str structure_dir: The directory containing the structure's micrographs and STAR file.
    :param int box_size: The box size in pixels for the cropped particles. If None, the function
                         will dynamically determine the box size from the .mrc map used for projections.
    :param float pixelsize: Pixel size of the micrographs.
    :param int max_crop_particles: The maximum number of particles to crop from the micrographs.
    :param int num_cpus: Number of CPU cores for parallel processing if GPUs are not used.
    :param bool use_gpu: Whether to use GPU for processing.
    :param list gpu_ids: List of GPU IDs to use for processing.
    :return int: Total number of particles cropped from the micrographs.

    Notes:
    - Particles whose specified box would extend beyond the micrograph boundaries are not cropped.
    - This function assumes the presence of a STAR file in the structure directory with the naming
      convention of '{structure_name}.star' or {structure_dir}/{structure_dir}.star containing the
      necessary coordinates for cropping.
    """
    print_and_log("", logging.DEBUG)
    print_and_log(f"\n[{structure_dir}] Cropping particles...\n")
    particles_dir = os.path.join(structure_dir, 'Particles/')
    os.makedirs(particles_dir, exist_ok=True)

    star_file_path = next((path for path in [f'{structure_dir}/{structure_dir}.star', f'{structure_dir}.star'] if os.path.isfile(path)), None)  # Hack to get around a bug where the star file is either in the base directory or structure directory.
    df = read_star_particles(star_file_path)
    df['particle_counter'] = range(1, len(df) + 1)

    grouped_df = df.groupby('micrograph_name')
    total_cropped = 0

    if use_gpu and gpu_ids:
        batch_sizes = {}
        for gpu_id in gpu_ids:
            utilization = get_gpu_utilization(gpu_id)
            batch_size = get_max_batch_size(box_size * box_size * 4, utilization['free_mem'])  # * 4 is because each pixel takes 4 bytes for float32
            batch_sizes[gpu_id] = {'batch_size': batch_size, 'core_usage': utilization['core_usage']}

        micrograph_list = list(grouped_df.groups.keys())
        start = 0
        while start < len(micrograph_list):
            sorted_gpus = sorted(batch_sizes.items(), key=lambda x: x[1]['core_usage'])
            for gpu_id, batch_info in sorted_gpus:
                end = start + batch_info['batch_size']
                batch_micrographs = micrograph_list[start:end]
                if not batch_micrographs:
                    break
                with cp.cuda.Device(gpu_id):
                    batch_paths = []
                    batch_particle_rows = []
                    for micrograph_name in batch_micrographs:
                        particle_rows = grouped_df.get_group(micrograph_name)
                        micrograph_path = os.path.join(micrograph_name)
                        if not os.path.exists(micrograph_path):
                            print_and_log(f"Micrograph not found: {micrograph_path}", logging.WARNING)
                            continue
                        print_and_log(f"[{structure_dir}] Extracting {len(particle_rows)} particles for {micrograph_name}...")
                        batch_paths.append(micrograph_path)
                        batch_particle_rows.append(particle_rows)
                    total_cropped += crop_particles_gpu_batch(batch_paths, batch_particle_rows, particles_dir, box_size, pixelsize, max_crop_particles, gpu_id)
                start = end
                if start >= len(micrograph_list):
                    break
    else:
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = []
            for micrograph_name, particle_rows in grouped_df:
                micrograph_path = os.path.join(micrograph_name)
                if not os.path.exists(micrograph_path):
                    print_and_log(f"Micrograph not found: {micrograph_path}", logging.WARNING)
                    continue

                print_and_log(f"[{structure_dir}] Extracting {len(particle_rows)} particles for {micrograph_name}...")
                future = executor.submit(crop_particles, micrograph_path, particle_rows, particles_dir, box_size, pixelsize, max_crop_particles)
                futures.append(future)

            for future in futures:
                total_cropped += future.result()

    return total_cropped

def process_single_micrograph(args, structure_name, structure, line, total_structures, structure_index,
                              micrograph_usage_count, ice_scaling_fudge_factor, remaining_aggregation_amounts, micrograph_number):
    """
    Process a single micrograph for a given structure.

    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :param str structure_name: The structure name from which synthetic micrographs are to be generated.
    :param numpy_array structure: The 3D numpy array of the structure.
    :param str line: The line containing the micrograph name and defocus value.
    :param int total_structures: Total number of structures requested.
    :param int structure_index: Index of the current structure.
    :param dict micrograph_usage_count: Dictionary to keep track of the usage count of each unique micrograph.
    :param float ice_scaling_fudge_factor: Fudge factor for making particles look dark enough.
    :param list remaining_aggregation_amounts: List to track remaining aggregation amounts.
    :param int micrograph_number: Current micrograph number (1-indexed).
    :return int int: Number of particles projected, number of particles with saved coordinates.
    """
    print_and_log("", logging.DEBUG)
    # Parse the 'micrograph_name.mrc defocus' line
    fname, defocus = line.strip().split()[:2]
    fname = os.path.splitext(os.path.basename(fname))[0]
    micrograph = readmrc(f"{args.image_directory}/{fname}.mrc")
    
    # Track each micrograph's usage count for naming purposes
    micrograph_usage_count[fname] = micrograph_usage_count.get(fname, 0) + 1
    # Generate the repeat number suffix for the filename
    repeat_suffix = f"_{micrograph_usage_count[fname]}" if micrograph_usage_count[fname] > 1 else ""

    # Add context for printout
    context = f"[{structure_name} | {micrograph_number}/{args.num_images}]"

    num_hyphens = '-' * (55 + len(f"{structure_index + 1}{total_structures}{structure_name}{fname}"))
    print_and_log(f"\n\033[1m{num_hyphens}\033[0m", logging.WARNING)
    print_and_log(f"Generating synthetic micrograph #{micrograph_number} using {structure_name} ({structure_index + 1}/{total_structures}) from {fname}...", logging.WARNING)
    print_and_log(f"\033[1m{num_hyphens}\033[0m\n", logging.WARNING)

    # Determine if overlap is allowed for this micrograph
    if args.allow_overlap_random:
        args.allow_overlap = bool(random.getrandbits(1))
    print_and_log(f"{context} {'Allowing' if args.allow_overlap else 'Not allowing'} overlapping particles for this micrograph.")

    # Determine ice and particle behavior parameters
    ice_thickness, ice_thickness_printout, num_particles, dist_type, non_random_dist_type, aggregation_amount_val = determine_ice_and_particle_behavior(
        args, structure, micrograph, ice_scaling_fudge_factor, remaining_aggregation_amounts, context)

    # Generate projections with the specified orientation mode
    print_and_log(f"{context} Projecting the structure volume ({structure_name}) {num_particles} times...")
    particles, orientations = generate_projections(structure, num_particles, args.orientation_mode, args.preferred_angles,
                                                   args.angle_variation, args.preferred_weight, args.cpus, args.use_gpu, args.gpu_ids)

    print_and_log(f"{context} Simulating pixel-level Poisson noise{f' and dose damage' if args.dose_damage != 'None' else ''} across {args.num_simulated_particle_frames} particle frame{'s' if args.num_simulated_particle_frames != 1 else ''}...")
    mean, gaussian_variance = estimate_noise_parameters(micrograph)
    if args.use_gpu:
        noisy_particles = add_poisson_noise_gpu(particles, args.num_simulated_particle_frames, args.dose_a, args.dose_b,
                                                args.dose_c, args.apix, args.gpu_ids)
    else:
        noisy_particles = add_poisson_noise(particles, args.num_simulated_particle_frames, args.dose_a, args.dose_b,
                                            args.dose_c, args.apix, args.cpus)

    print_and_log(f"{context} Applying CTF based on defocus ({float(defocus):.4f} microns) and microscope parameters ({args.voltage} keV, AmpCont: {args.ampcont}%, Cs: {args.Cs} mm, Pixelsize: {args.apix} Angstroms) of the ice micrograph...")
    noisy_particles_CTF = apply_ctfs_with_eman2(noisy_particles, [defocus] * len(noisy_particles), args.ampcont, args.bfactor,
                                                args.apix, args.Cs, args.voltage, args.cpus)

    print_and_log(f"{context} Adding {num_particles} particles to the micrograph{f' {dist_type}ly ({non_random_dist_type})' if dist_type == 'non_random' else f' {dist_type}ly' if dist_type else ''}{f' with aggregation amount of {aggregation_amount_val:.1f}' if args.distribution in ('m','micrograph') else ''} while adding Gaussian (white) noise and simulating a average relative ice thickness of {ice_thickness_printout:.1f} nm...")

    # Make dictionaries of parameters to pass to make it easy to add/change parameters with continued development
    input_options = {
        'large_image_path': f"{args.image_directory}/{fname}.mrc",
        'large_image': micrograph,
        'small_images': noisy_particles_CTF,
        'pixelsize': args.apix,
        'structure_name': structure_name,
        'orientations': orientations
    }
    particle_and_micrograph_generation_options = {
        'scale_percent': args.scale_percent,
        'dist_type': dist_type,
        'non_random_dist_type': non_random_dist_type,
        'border_distance': args.border,
        'no_edge_particles': args.no_edge_particles,
        'save_edge_coordinates': args.save_edge_coordinates,
        'gaussian_variance': gaussian_variance,
        'aggregation_amount': aggregation_amount_val,
        'allow_overlap': args.allow_overlap,
        'save_overlapping_coords': args.save_overlapping_coords
    }
    simulation_options = {'scale': ice_thickness}
    junk_labels_options = {
        'no_junk_filter': args.no_junk_filter,
        'flip_x': args.flip_x,
        'flip_y': args.flip_y,
        'json_scale': args.json_scale,
        'polygon_expansion_distance': args.polygon_expansion_distance
    }
    output_options = {
        'save_as_mrc': args.mrc,
        'save_as_png': args.png,
        'save_as_jpeg': args.jpeg,
        'jpeg_quality': args.jpeg_quality,
        'imod_coordinate_file': args.imod_coordinate_file,
        'coord_coordinate_file': args.coord_coordinate_file,
        'output_path': f"{structure_name}/{fname}_{structure_name}{repeat_suffix}"
    }

    num_particles_projected, num_particles_with_saved_coordinates = add_images(
    input_options, particle_and_micrograph_generation_options,
    simulation_options, junk_labels_options, output_options, context, defocus)

    return num_particles_projected, num_particles_with_saved_coordinates

def generate_micrographs(args, structure_name, structure_type, structure_index, total_structures):
    """
    Generate synthetic micrographs for a specified structure.

    This function orchestrates the workflow for generating synthetic micrographs
    for a given structure. It performs file download, conversion, image shuffling,
    and iterates through the generation process for each selected image. It also
    handles cleanup operations post-generation.

    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :param str structure_name: The structure name from which synthetic micrographs are to be generated.
    :param str structure_type: Whether the structure is a pdb or mrc.
    :param str structure_index: The index of the structure that micrographs are being generated from.
    :param str total_structures: The total number of structures requested.

    :return int int: Total number of particles actually added to all micrographs, and box size of the projected volume.

    This function writes .mrc and .txt files to the disk and deletes several files during cleanup.
    """
    print_and_log("", logging.DEBUG)
    os.makedirs(structure_name, exist_ok=True)  # Create output directory

    # Convert PDB to MRC for PDBs and return the mass or estimate the mass of the MRC
    if structure_type == "pdb":
        mass = convert_pdb_to_mrc(structure_name, args.apix, args.pdb_to_mrc_resolution)
        structure = reorient_mrc(f'{structure_name}.mrc')
        print_and_log(f"[{structure_name}] Reoriented MRC")
        print_and_log(f"[{structure_name}] Mass of PDB: {mass} kDa")
        ice_scaling_fudge_factor = 4.8  # Note: larger number = darker particles
    elif structure_type == "mrc":
        mass = int(estimate_mass_from_map(structure_name))
        if args.reorient_mrc:
            structure = reorient_mrc(f'{structure_name}.mrc')  # It's very slow for EMDB maps...
            print_and_log(f"[{structure_name}] Reoriented MRC")
        else:
            structure = readmrc(f'{structure_name}.mrc')
        print_and_log(f"[{structure_name}] Estimated mass of MRC: {mass} kDa")
        ice_scaling_fudge_factor = 2.9

    # Write STAR header for the current synthetic dataset
    write_star_header(structure_name, args.apix, args.voltage, args.Cs)

    # Shuffle and possibly extend the ice images
    print_and_log(f"[{structure_name}] Selecting {args.num_images} random ice micrographs...")
    selected_images = extend_and_shuffle_image_list(args.num_images, args.image_list_file)

    # Main Loop
    total_num_particles_projected = 0
    total_num_particles_with_saved_coordinates = 0
    micrograph_usage_count = {}  # Dictionary to keep track of repeating micrograph names if the image list was extended
    remaining_aggregation_amounts = list()

    # Check if parallelization is requested
    if args.parallelize_micrographs > 1:
        # Use ProcessPoolExecutor to parallelize the micrograph generation
        with ProcessPoolExecutor(max_workers=args.parallelize_micrographs) as executor:
            # Create a list to store the future tasks
            futures = []
            
            # Iterate over the selected_images
            for i, line in enumerate(selected_images):
                # Submit each micrograph generation task to the executor
                future = executor.submit(process_single_micrograph, args, structure_name, structure, line, total_structures,
                                         structure_index, micrograph_usage_count, ice_scaling_fudge_factor, remaining_aggregation_amounts,
                                         i + 1)  # Pass the current micrograph number (1-indexed)
                futures.append(future)
            
            # Iterate over the completed tasks
            for future in futures:
                # Aggregate the results from each task
                num_particles_projected, num_particles_with_saved_coordinates = future.result()
                total_num_particles_projected += num_particles_projected
                total_num_particles_with_saved_coordinates += num_particles_with_saved_coordinates
    else:
        # If no parallelization is requested, process each micrograph sequentially
        for i, line in enumerate(selected_images):
            num_particles_projected, num_particles_with_saved_coordinates = process_single_micrograph(
                args, structure_name, structure, line, total_structures, structure_index, micrograph_usage_count,
                ice_scaling_fudge_factor, remaining_aggregation_amounts, i + 1)  # Pass the current micrograph number (1-indexed)
            total_num_particles_projected += num_particles_projected
            total_num_particles_with_saved_coordinates += num_particles_with_saved_coordinates

    # Downsample micrographs and coordinate files
    if args.binning > 1:
        print_and_log(f"[{structure_name}] Binning/Downsampling micrographs by {args.binning} by Fourier cropping...")
        parallel_downsample_micrographs(f"{structure_name}/", args.binning, args.apix, args.cpus, args.use_gpu, args.gpu_ids)
        downsample_coordinate_files(structure_name, args.binning, args.imod_coordinate_file, args.coord_coordinate_file)
    else:
        shutil.move(f"{structure_name}.star", f"{structure_name}/")

    # Log structure name, mass, number of micrographs generated, number of particles projected, and number of particles written to coordinate files
    with open(f"virtualice_{args.script_start_time}_info.txt", "a") as f:
        # Check if the file is empty
        if f.tell() == 0:  # Only write the first line if the file is new
            f.write("structure_name mass(kDa) num_images num_particles_projected num_particles_saved\n")

        # Write the subsequent lines
        f.write(f"{structure_name} {mass} {args.num_images} {total_num_particles_projected} {total_num_particles_with_saved_coordinates}\n")

    box_size = structure.shape[0]  # Assuming the map is a cube

    return total_num_particles_projected, total_num_particles_with_saved_coordinates, box_size

def clean_up(args, structure_name):
    """
    Clean up files at the end of the script.

    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :param str structure_name: The structure name from which synthetic micrographs were generated.

    This function deletes several files.
    """
    print_and_log("", logging.DEBUG)
    if args.binning > 1:
        if not args.keep:
            print_and_log(f"[{structure_name}] Removing non-downsamlpled micrographs...")
            bin_dir = f"{structure_name}/bin_{args.binning}/"

            # Delete the non-downsampled micrographs and coordinate files
            file_extensions = ["mrc", "png", "jpeg", "mod", "coord"]
            for ext in file_extensions:
                for file in glob.glob(f"{structure_name}/*.{ext}"):
                    os.remove(file)

            # Move the binned files to the parent directory
            for ext in file_extensions:
                for file in glob.glob(f"{bin_dir}/*.{ext}"):
                    shutil.move(file, f"{structure_name}/")

            shutil.rmtree(bin_dir)
            shutil.move(f"{structure_name}_bin{args.binning}.star", f"{structure_name}/")
            os.remove(f"{structure_name}.star")
        else:
            shutil.move(f"{structure_name}.star", f"{structure_name}/")
            shutil.move(f"{structure_name}_bin{args.binning}.star", f"{structure_name}/bin_{args.binning}/")

    for directory in [f"{structure_name}/", f"{structure_name}/bin_{args.binning}/"]:
        try:
            for file_name in os.listdir(directory):
                if file_name.endswith(".point"):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)
        except FileNotFoundError:
            pass

    for temp_file in [f"{structure_name}.pdb", f"{structure_name}.map", f"{structure_name}.mrc", "thread.out", ".eman2log.txt"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def print_run_information(num_micrographs, structure_names, time_str, total_number_of_particles_projected,
                          total_number_of_particles_with_saved_coordinates, total_cropped_particles, crop_particles,
                          imod_coordinate_file, coord_coordinate_file, output_directory):
    """
    Print run information based on the provided inputs.

    :param int num_micrographs: The number of micrographs.
    :param list structure_names: List of names of all of the structures.
    :param str time_str: The string representation of the total generation time.
    :param int total_number_of_particles_projected: The total number of particles projected.
    :param int total_number_of_particles_with_saved_coordinates: Total number of particles saved to coordinate files.
    :param int total_cropped_particles: The total number of cropped particles.
    :param bool crop_particles: Whether or not particles were cropped.
    :param bool imod_coordinate_file: Whether to downsample and save IMOD .mod coordinate files.
    :param bool coord_coordinate_file: Whether to downsample and save .coord coordinate files.
    :param str output_directory: Output directory for all run files.
    """
    print_and_log("", logging.DEBUG)
    total_structures = len(structure_names)
    print_and_log(f"\n\n\033[1m{'-' * 100}\n{('VirtualIce Generation Summary').center(100)}\n{'-' * 100}\033[0m", logging.WARNING)
    print_and_log(f"Time to generate \033[1m{num_micrographs}\033[0m micrograph{'s' if num_micrographs != 1 else ''} from \033[1m{total_structures}\033[0m structure{'s' if total_structures != 1 else ''}: \033[1m{time_str}\033[0m", logging.WARNING)
    print_and_log(f"Total: \033[1m{total_number_of_particles_projected}\033[0m particles projected, \033[1m{total_number_of_particles_with_saved_coordinates}\033[0m saved to {'.star file' if not imod_coordinate_file and not coord_coordinate_file else 'coordinate files'}" + (f", \033[1m{total_cropped_particles}\033[0m particles cropped" if crop_particles else ""), logging.WARNING)
    print_and_log(f"Run directory: \033[1m{output_directory}/\033[0m" + (f", Crop sub-directory: \033[1m{structure_names[0] if total_structures == 1 else '[structure_names]'}/Particles/\033[0m\n" if crop_particles else "\n"), logging.WARNING)

    print_and_log(f"One .star file is located in {'the' if total_structures == 1 else 'each'} structure sub-directory.", logging.WARNING)

    if coord_coordinate_file:
        print_and_log(f"One (x y) .coord file per micrograph is located in the structure sub-director{'y' if total_structures == 1 else 'ies'}.", logging.WARNING)

    if imod_coordinate_file:
        print_and_log(f"One IMOD .mod file per micrograph is located in the structure sub-director{'y' if total_structures == 1 else 'ies'}.", logging.WARNING)
        print_and_log("To open a micrograph with an IMOD .mod file, run a command of this form:", logging.WARNING)
        print_and_log("  \033[1m3dmod image.mrc image.mod\033[0m  (Replace with your files)", logging.WARNING)
    print_and_log(f"\033[1m{'-' * 100}\033[0m\n", logging.WARNING)

def find_micrograph_files(structure_dirs):
    """
    Find all micrograph files (.mrc, .png, .jpeg) in the given structure directories.
    Order final list so that for a given micrograph, the mrc file is opened first, then png, then jpeg.

    :param list structure_dirs: List of structure directories to search for micrograph files.
    :return list: List of paths to found micrograph files.
    """
    print_and_log("", logging.DEBUG)
    micrograph_files = []
    for structure_dir in structure_dirs:
        micrograph_files.extend(glob.glob(os.path.join(structure_dir, '*.mrc')))
        micrograph_files.extend(glob.glob(os.path.join(structure_dir, '*.png')))
        micrograph_files.extend(glob.glob(os.path.join(structure_dir, '*.jpeg')))

    extension_order = {'.mrc': 0, '.png': 1, '.jpeg': 2}
    micrograph_files = sorted(micrograph_files, key=lambda f: (f.rsplit('.', 1)[0], extension_order.get('.' + f.rsplit('.', 1)[1], 3)))
    return micrograph_files

def view_in_3dmod_async(micrograph_files, imod_coordinate_file):
    """
    Open micrograph files in 3dmod asynchronously.

    :param list micrograph_files: List of micrograph file paths to open in 3dmod.
    :param bool imod_coordinate_file: Whether IMOD .mod coordinate files were saved.
    """
    print_and_log("", logging.DEBUG)
    if len(micrograph_files) == 1 and imod_coordinate_file:  # Open the micrograph and coordinates if there's only 1 micrograph
        mod_file = [micrograph_files[0].rsplit('.', 1)[0] + '.mod']
        subprocess.run(["3dmod"] + micrograph_files + mod_file)
    else:
        subprocess.run(["3dmod"] + micrograph_files)
    print_and_log("Opening micrographs in 3dmod...\n")

def main():
    """
    Main function: Loops over structures, generates micrographs, optionally crops particles,
    optionally opens micrographs in 3dmod asynchronously, and prints run time & output information.
    """
    start_time = time.time()
    start_time_formatted = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))

    # Parse user arguments, check them, and update them conditionally if necessary
    args = parse_arguments(start_time_formatted)

    # Loop over each provided structure and generate micrographs. Skip a structure if it doesn't download/exist
    total_structures = len(args.structures)
    total_number_of_particles_projected = 0
    total_number_of_particles_with_saved_coordinates = 0
    total_cropped_particles = 0
    with ProcessPoolExecutor(max_workers=args.parallelize_structures) as executor:
        # Store structure_names and tasks (ie. results)
        structure_names = []
        tasks = []
        for structure_index, structure_input in enumerate(args.structures):
            result = process_structure_input(structure_input, args.max_emdb_size, args.std_threshold, args.apix)
            if result:  # Check if result is not None
                structure_name, structure_type = result  # Now we're sure structure_name and structure_type are valid
                task = executor.submit(generate_micrographs, args, structure_name, structure_type, structure_index, total_structures)
                tasks.append((task, structure_name))  # Store task and associated structure_name
            else:
                print_and_log(f"Skipping structure (error or non-existence): {structure_input} (use `-s r` for a random structure)", logging.WARNING)

        # Wait for all tasks to complete and aggregate results
        for task, structure_name in tasks:
            structure_names.append(structure_name)
            num_particles_projected, num_particles_with_saved_coordinates, box_size = task.result()
            total_number_of_particles_projected += num_particles_projected
            total_number_of_particles_with_saved_coordinates += num_particles_with_saved_coordinates

    # Open 3dmod asynchronously in a separate thread so cropping can happen simultaneously
    if args.view_in_3dmod:
        micrograph_files = find_micrograph_files(structure_names)
        if micrograph_files:
            threading.Thread(target=view_in_3dmod_async, args=(micrograph_files, args.imod_coordinate_file)).start()

    # Crop particles if requested
    if args.crop_particles:
        with ProcessPoolExecutor(max_workers=args.parallelize_structures) as executor:
            crop_tasks = []
            for structure_name in structure_names:
                box_size = args.box_size if args.box_size is not None else box_size
                crop_task = executor.submit(crop_particles_from_micrographs, structure_name, box_size, args.apix, args.max_crop_particles, args.cpus, args.use_gpu, args.gpu_ids)
                crop_tasks.append(crop_task)

            for crop_task in crop_tasks:
                total_cropped_particles += crop_task.result()

    for structure_name in structure_names:
        clean_up(args, structure_name)

    num_micrographs = args.num_images * len(structure_names)
    end_time = time.time()
    time_str = time_diff(end_time - start_time)

    print_run_information(num_micrographs, structure_names, time_str, total_number_of_particles_projected,
                          total_number_of_particles_with_saved_coordinates, total_cropped_particles, args.crop_particles,
                          args.imod_coordinate_file, args.coord_coordinate_file, args.output_directory)

if __name__ == "__main__":
    main()
