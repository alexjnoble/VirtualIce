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
__version__ = "2.0.0"

import os
import re
import ast
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
from scipy.spatial import KDTree
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
            raise argparse.ArgumentTypeError("Number of particles must be between 2 and 1000000 or 'max'")
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
            raise argparse.ArgumentTypeError("Binning must be an integer between 2 and 64")
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError("Binning must be an integer between 2 and 64")

def format_structure_sets(structure_sets):
    """
    Formats a list of structure sets for printing in the run configuration.

    Each structure set can contain one or more structures. The function formats
    each structure set as a string for display purposes, showing single structures
    without brackets and multiple structures enclosed in brackets.

    Example:
    - Input: [['1PMA'], ['1TIM']]
      Output: "1PMA, 1TIM"

    - Input: [[['1PMA'], ['1TIM']], [['my_structure.mrc'], ['rp']]]
      Output: "[1PMA, 1TIM], [my_structure.mrc, rp]"

    :param list_of_lists structure_sets: A list of structure sets, where each set is a list of structure IDs.
    :return list_of_str: A list of formatted structure sets as strings.
    """
    print_and_log("", logging.DEBUG)
    formatted_sets = []

    # Iterate over each structure set
    for structure_set in structure_sets:
        # Check if the structure_set contains multiple structures
        if isinstance(structure_set[0], list):
            # Flatten the list of lists and join them with commas
            flat_set = [s[0] for s in structure_set]  # Extract strings from the inner lists
            formatted_sets.append(f"[{', '.join(flat_set)}]")
        else:
            # If it's a single structure, just add it as is (without brackets)
            formatted_sets.append(structure_set[0])

    return formatted_sets

def parse_structure_input(structure_input):
    """
    Parses the structure input argument into a list of structure sets.

    Each structure set can contain one or more structures, and each structure set will be applied 
    to generate a corresponding set of micrographs. The input can handle both single-structure-per-micrograph 
    and multiple-structures-per-micrograph formats in the forms:

    1. Single structures per micrograph. Example: ['1TIM', 'mystructure.mrc', 'rp']
         This will be outputted as a list of lists: [['1PMA'], ['1TIM']]

    2. Multiple structures per micrograph. Example: ['[1PMA,', '1TIM]', '[my_structure2.mrc,', 'rp]']
         This will be outputted as a list of lists of lists: [[['1PMA'], ['1TIM']], [['my_structure2.mrc'], ['rp']]]

    :param list_of_str structure_input: List of structure input arguments from the command line.
    :return list_of_lists or list_of_lists_of_lists: A list of structure sets, where each set is a list of structure IDs.
    :raises argparse.ArgumentTypeError: If the structure input format is invalid.
    """
    print_and_log("", logging.DEBUG)
    structure_sets = []
    current_set = []
    in_bracket = False

    for item in structure_input:
        if item.startswith('['):
            if in_bracket:
                raise argparse.ArgumentTypeError(f"Nested brackets are not allowed: {item}")
            in_bracket = True
            current_set = []
            item = item[1:]  # Remove opening bracket

        if item.endswith(']'):
            if not in_bracket:
                raise argparse.ArgumentTypeError(f"Closing bracket without opening bracket: {item}")
            in_bracket = False
            item = item[:-1]  # Remove closing bracket

        structures = [s.strip() for s in item.split(',') if s.strip()]
        current_set.extend([[s] for s in structures])

        if not in_bracket:
            if len(current_set) > 1:
                structure_sets.append(current_set)
            else:
                structure_sets.extend(current_set)
            current_set = []

    if in_bracket:
        raise argparse.ArgumentTypeError("Unclosed bracket in input")

    return structure_sets

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
        # Handle the case where values is 'r' or 'random'
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

def convert_color_to_rgb(color):
    """
    Convert color name or RGB values to a tuple of RGB integers.

    :param str or list color: Color name or list of RGB values.
    :return tuple: RGB values as integers (0-255).
    """
    print_and_log("", logging.DEBUG)
    color_map = dict(
        red=(255,0,0), green=(0,255,0), blue=(0,0,255),
        yellow=(255,255,0), cyan=(0,255,255), magenta=(255,0,255),
        white=(255,255,255), black=(0,0,0), gray=(128,128,128),
        orange=(255,165,0), purple=(128,0,128), pink=(255,192,203),
        brown=(165,42,42), navy=(0,0,128), teal=(0,128,128),
        maroon=(128,0,0), olive=(128,128,0), lime=(0,255,0),
        aqua=(0,255,255), silver=(192,192,192), indigo=(75,0,130),
        violet=(238,130,238), turquoise=(64,224,208), coral=(255,127,80),
        gold=(255,215,0), salmon=(250,128,114), khaki=(240,230,140),
        plum=(221,160,221), crimson=(220,20,60), lavender=(230,230,250),
        beige=(245,245,220), ivory=(255,255,240), mint=(189,252,201),
        forest_green=(34,139,34), royal_blue=(65,105,225), dark_red=(139,0,0),
        sky_blue=(135,206,235), hot_pink=(255,105,180), sea_green=(46,139,87),
        steel_blue=(70,130,180), sienna=(160,82,45), tan=(210,180,140),
        dark_violet=(148,0,211), firebrick=(178,34,34), midnight_blue=(25,25,112),
        rosy_brown=(188,143,143), light_coral=(240,128,128),
    )

    if color == 'random' or color == 'r':
        return random.choice(list(color_map.values()))
    elif isinstance(color, str):
        color = color.lower()
        if color in color_map:
            return color_map[color]
        else:
            raise ValueError(f"Unknown color name: {color}")
    elif isinstance(color, list) and len(color) == 3:
        return tuple(map(int, color))
    else:
        raise ValueError("Color must be a valid color name, a list of 3 RGB values (0-255), or 'random'")

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

def parse_arguments(script_start_time):
    """
    Parses command-line arguments.

    :param str script_start_time: Function start time, formatted as a string
    :returns argparse.Namespace: An object containing attributes for each command-line argument.
    """
    # Determine the installation directory
    installation_directory = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="\033[1mVirtualIce:\033[0m A feature-rich synthetic cryoEM micrograph generator that projects pdbs|mrcs onto existing buffer cryoEM micrographs. Star files for particle coordinates are outputed by default, mod and coord files are optional. Particle coordinates located within per-micrograph polygons at junk/substrate locations are projected but not written to coordinate files.",
    epilog="""
\033[1mExamples:\033[0m
  1. Basic usage: virtualice.py -s 1TIM -n 10
     Generates 10 random micrographs of PDB 1TIM (single-structure micrographs).

  2. Basic usage: virtualice.py -s [1TIM, 11638] 1PMA -n 10
     Generates 10 random micrographs for the structure set consisting of PDB 1TIM and EMDB-11638 (multi-structure micrographs),
     and 10 random micrographs of PDB 1PMA (single-structure micrographs).

  3. Advanced usage: virtualice.py -s 1TIM r my_structure.mrc 11638 rp -n 3 -I -P -J -Q 90 -b 4 -D n -ps 2 -C
     Generates 3 random micrographs of PDB 1TIM, a random EMDB/PDB structure, a local structure called my_structure.mrc, EMD-11638, and a random PDB.
     Outputs an IMOD .mod coordinate file, png, and jpeg (quality 90) for each micrograph, and bins all images by 4.
     Uses a non-random distribution of particles, parallelizes structure generation across 2 CPUs, and crops particles.

  4. Advanced usage: virtualice.py -s 1PMA -n 5 -om preferred -pw 0.9 -pa [*,90,0] [90 180 *] -aa l h r -ne --use_cpu -V 2 -3
     Generates 5 random micrographs of PDB 1PMA (proteasome) with preferred orientation for 90% of particles. The preferred orientations are defined
     by random selections of [*,90,0] (free to rotate along the first Z axis, then rotate 90 degrees in Y, do not rotate in Z) and
     [90 180 0] (rotate 90 degrees along the first Z axis, then rotate 180 degrees in Y, then free to rotate along the resulting Z). The aggregation amount is
     randomly chosen from low and high values for each of the 5 micrographs. Edge particles are not included. Only CPUs are used (no GPUs).
     Terminal verbosity is set to 2. The resulting micrographs are opened with 3dmod after generation.
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter)  # Preserves whitespace for better formatting

    # Input Options
    input_group = parser.add_argument_group('\033[1mInput Options\033[0m')
    input_group.add_argument("-s", "--structures", type=str, nargs='+', default=['1TIM', '19436', 'r'], help="PDB ID(s), EMDB ID(s), names of local .pdb or .mrc/.map files, 'r' or 'random' for a random PDB or EMDB map, 'rp' for a random PDB, and/or 're' or 'rm' for a random EMDB map. Local .mrc/.map files must have voxel size in the header so that they are scaled properly. To specify multiple structures per micrograph, use brackets: [structure1, structure2]. Example: -s [my_structure1.mrc, 1TIM, rp] [my_structure2.mrc, 12345, re] or single structures like 1TIM my_structure.mrc rp. Note: PDB files are recommended because noise levels of .mrc/.map files are unpredictable. Default is %(default)s.")
    input_group.add_argument("-d", "--image_directory", type=str, default=os.path.join(installation_directory, "ice_images"), help="Local directory name where the micrographs are stored in mrc format. They need to be accompanied with a text file containing image names and defoci (see --image_list_file). Default directory is %(default)s")
    input_group.add_argument("-i", "--image_list_file", type=str, default=os.path.join(installation_directory, "ice_images/good_images_with_defocus.txt"), help="File containing local filenames of images with a defocus value after each filename (space between). Default is '%(default)s'.")
    input_group.add_argument("-me", "--max_emdb_size", type=float, default=512, help="The maximum allowed file size in megabytes. Default is %(default)s")

    # Particle and Micrograph Generation Options
    particle_micrograph_group = parser.add_argument_group('\033[1mParticle and Micrograph Generation Options\033[0m')
    particle_micrograph_group.add_argument("-n", "--num_images", type=int, default=5, help="Number of micrographs to create for each structure requested. Default is %(default)s")
    particle_micrograph_group.add_argument("-N", "--num_particles", type=check_num_particles, help="Number of particles to project onto the micrograph after rotation. Input an integer or 'max'. Default is a random number (weighted to favor numbers above 100 twice as much as below 100) up to a maximum of the number of particles that can fit into the micrograph without overlapping.")
    particle_micrograph_group.add_argument("-a", "--apix", type=float, default=1.096, help="Pixel size (in Angstroms) of the ice images, used to scale pdbs during pdb>mrc conversion (EMAN2 e2pdb2mrc.py option). Default is %(default)s (the pixel size of the ice images used during development)")
    particle_micrograph_group.add_argument("-r", "--pdb_to_mrc_resolution", type=float, default=3, help="Resolution in Angstroms for PDB to MRC conversion (EMAN2 e2pdb2mrc.py option). Default is %(default)s")
    particle_micrograph_group.add_argument("-st", "--std_threshold", type=float, default=-1.0, help="Threshold for removing noise from a downloaded/imported .mrc/.map file in terms of standard deviations above the mean. The idea is to not have dust around the 3D volume from the beginning. Default is %(default)s")
    particle_micrograph_group.add_argument("-sp", "--scale_percent", type=float, default=33.33, help="How much larger to make the resulting mrc file from the pdb file compared to the minimum equilateral cube. Extra space allows for more delocalized CTF information (default: %(default)s; ie. %(default)s%% larger)")
    particle_micrograph_group.add_argument("-D", "--distribution", type=str, choices=['m', 'micrograph', 'g', 'gaussian', 'c', 'circular', 'ic', 'inverse_circular', 'r', 'random', 'n', 'non_random'], default='micrograph', help="Distribution type for generating particle locations: 'random' (or 'r') and 'non_random' (or 'n'). random is a random selection from a uniform 2D distribution. non_random selects from 4 distributions that can alternatively be requested directly: 1) 'micrograph' (or 'm') to mimic ice thickness (darker areas = more particles), 2) 'gaussian' (or 'g') clumps, 3) 'circular' (or 'c'), and 4) 'inverse_circular' (or 'ic'). Default is %(default)s which selects a distribution per micrograph based on internal weights.")
    particle_micrograph_group.add_argument("-aa", "--aggregation_amount", nargs='+', default=['low', 'random'], help="Amount of particle aggregation. Aggregation amounts can be set per-run or per-micrograph. To set per-run, input 0-10, 'low', 'medium', 'high', or 'random'. To set multiple aggregation amounts that will be chose randomly per-micrograph, input combinations like 'low medium', 'low high', '2 5', or 'low 3 9 10'. To set random aggregation amounts within a range, append any word input combination with 'random', like 'random random' to give the full range, or 'low medium random' to give a range from 0 to 6.7. Abbreviations work too, like '3.2 l h r'. Default is %(default)s")
    particle_micrograph_group.add_argument("-ao", "--allow_overlap", type=str, choices=['True', 'False', 'random'], default='random', help="Specify whether to allow overlapping particles. Options are 'True', 'False', or 'random'. Default is %(default)s")
    particle_micrograph_group.add_argument("-ado", "--allowed_overlap", type=float, default=0.92, help="A factor related to how close non-overlapping particles can be when determining whether tosave their coordinates or not. Smaller = closer. Default is %(default)s, which was heuristically determined to work for globular proteins.")
    particle_micrograph_group.add_argument("-nl", "--num_particle_layers", type=int, default=2, help="If overlapping particles is allowed, this is the number of overlapping particle layers allowed (soft condition, not strict. Used in determining the maximum number of particles that can be placed in a micrograph). Default is %(default)s")
    particle_micrograph_group.add_argument("-so", "--save_overlapping_coords", action="store_true", help="Save overlapping particle coordinates to output files. Default is to not save overlapping particle")
    particle_micrograph_group.add_argument("-B", "--border", type=int, default=-1, help="Minimum distance of center of particles from the image border. Default is %(default)s = reverts to half boxsize")
    particle_micrograph_group.add_argument("-ne", "--no_edge_particles", action="store_true", help="Prevent particles from being placed up to the edge of the micrograph. By default, particles can be placed up to the edge.")
    particle_micrograph_group.add_argument("-se", "--save_edge_coordinates", action="store_true", help="Save particle coordinates that are closer than --border or closer than half a particle box size (if --border is not specified) from the edge. Requires --no_edge_particles to be False or --border to be greater than or equal to half the particle box size.")
    # TBD: Need to make a new border distance value for which particles are saved based on distance from the borders
    #particle_micrograph_group.add_argument("-sb", "--save_border", type=int, default=None, help="Minimum distance from the image border required to save a particle's coordinates to the output files. Default is %(default)s, which will use the value of --border if specified, otherwise half of the particle box size.")
    #particle_micrograph_group.add_argument("-soo", "--save_only_obscured_particles", action='store_true', help="Save only overlapping particles and within the junk masks.")

    # Simulation Options
    simulation_group = parser.add_argument_group('\033[1mSimulation Options\033[0m')
    simulation_group.add_argument("-dd", "--dose_damage", type=str, choices=['None', 'Light', 'Moderate', 'Heavy', 'Custom'], default='Moderate', help="Simulated protein damage due to accumulated dose, applied to simulated particle frames. Uses equation given by Grant & Grigorieff, 2015. Default is %(default)s")
    simulation_group.add_argument("-da", "--dose_a", type=float, required=False, help="Custom value for the \'a\' variable in equation (3) of Grant & Grigorieff, 2015 (only required if '--dose-preset Custom' is chosen).")
    simulation_group.add_argument("-db", "--dose_b", type=float, required=False, help="Custom value for the \'b\' variable in equation (3) of Grant & Grigorieff, 2015 (only required if '--dose-preset Custom' is chosen).")
    simulation_group.add_argument("-dc", "--dose_c", type=float, required=False, help="Custom value for the \'c\' variable in equation (3) of Grant & Grigorieff, 2015 (only required if '--dose-preset Custom' is chosen).")
    simulation_group.add_argument("-nf", "--num_simulated_particle_frames", type=int, default=50, help="Number of simulated particle frames to generate Poisson noise and optionally apply dose damaging. Default is %(default)s")
    simulation_group.add_argument("-m", "--min_ice_thickness", type=float, default=30, help="Minimum ice thickness, which scales how much the particle is added to the image (this is a relative value). Default is %(default)s")
    simulation_group.add_argument("-M", "--max_ice_thickness", type=float, default=150, help="Maximum ice thickness, which scales how much the particle is added to the image (this is a relative value). Default is %(default)s")
    simulation_group.add_argument("-t", "--ice_thickness", type=float, help="Request a specific ice thickness, which scales how much the particle is added to the image (this is a relative value). This will override --min_ice_thickness and --max_ice_thickness. Note: When choosing 'micrograph' particle distribution, the ice thickness uses the same gradient map to locally scale simulated ice thickness.")
    simulation_group.add_argument("-ro", "--reorient_mrc", action="store_true", help="Reorient the MRC file (either provided as a .mrc file or requested from EMDB) so that the structure's principal axes align with the coordinate axes. Note: .pdb files are automatically reoriented, EMDB files are often too big to do so quickly. Default is %(default)s")
    simulation_group.add_argument("-om", "--orientation_mode", type=str, choices=['random', 'uniform', 'preferred'], default='random', help="Orientation mode for projections. Options are: random, uniform, preferred. Default is %(default)s")
    simulation_group.add_argument("-pa", "--preferred_angles", type=str, nargs='+', default=None, help="List of sets of three Euler angles (in degrees) for preferred orientations. Use '*' as a wildcard for random angles. Example: '[90,0,0]' or '[*,0,90]'. Euler angles are in the range [0, 360] for alpha and gamma, and [0, 180] for beta. Default is %(default)s")
    simulation_group.add_argument("-av", "--angle_variation", type=float, default=5.0, help="Standard deviation for normal distribution of variations around preferred angles (in degrees). Default is %(default)s")
    simulation_group.add_argument("-pw", "--preferred_weight", type=float, default=0.9, help="Weight of the preferred orientations in the range [0, 1] (only used if orientation_mode is preferred). Default is %(default)s")
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
    output_group.add_argument("-k", "--keep", action="store_true", help="Keep the non-downsampled micrographs if downsampling is requested. Non-downsampled micrographs are deleted by default. Flag is currently forced to True until bug is fixed.")
    output_group.add_argument("-I", "--imod_coordinate_file", action="store_true", help="Also output one IMOD .mod coordinate file per micrograph. Note: IMOD must be installed and working")
    output_group.add_argument("-O", "--coord_coordinate_file", action="store_true", help="Also output one .coord coordinate file per micrograph")
    output_group.add_argument("-3", "--view_in_3dmod", action='store_true', help="View generated micrographs in 3dmod at the end of the run")
    output_group.add_argument("-3r", "--imod_circle_radius", type=int, default=10, help="Radius of the circle drawn in IMOD .mod files. Default is %(default)s")
    output_group.add_argument("-3t", "--imod_circle_thickness", type=int, default=2, help="Thickness of the circular line drawn in IMOD .mod files. Default is %(default)s")
    output_group.add_argument("-3c", "--imod_circle_color", nargs='+', default='green', help="Color of the circle drawn in IMOD .mod files. Can be a color name (e.g. 'red', 'green', 'blue'), three RGB values (0-255), or 'random'/'r' for a random color per structure. Default is %(default)s")

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

    args.structures = parse_structure_input(args.structures)

    if args.binning != 1:
        args.keep = True  # Hardcoded until bug is fixed.

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

    # Flatten args.structures to count the total number of structures across all sets
    flattened_structures = [structure for structure_set in args.structures for structure in structure_set]

    # Automatically adjust parallelization settings based on available CPU cores
    available_cpus = os.cpu_count()
    if args.parallelize_structures is None:
        # Parallelize based on the total number of structures across all sets
        args.parallelize_structures = min(max(1, available_cpus // 4), len(flattened_structures))
    else:
        args.parallelize_structures = min(args.parallelize_structures, len(flattened_structures))

    if args.parallelize_micrographs is None:
        args.parallelize_micrographs = min(max(1, available_cpus // 4), args.num_images)
    else:
        args.parallelize_micrographs = min(args.parallelize_micrographs, args.num_images)

    if args.cpus is None:
        args.cpus = max(1, available_cpus - args.parallelize_structures - args.parallelize_micrographs)

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

    if args.imod_circle_color == 'r' or args.imod_circle_color == 'random':
        args.imod_circle_color = 'random'
    elif len(args.imod_circle_color) == 1:
        args.imod_circle_color = args.imod_circle_color[0]  # It's a color name
    elif len(args.imod_circle_color) == 3:
        args.imod_circle_color = [int(v) for v in args.imod_circle_color]  # It's RGB values
    else:
        raise ValueError("--imod_circle_color must be either a color name, exactly 3 RGB values, or 'random'/'r'")

    # Print all arguments for the user's information
    formatted_output = ""
    for arg, value in vars(args).items():
        formatted_output += f"{arg}: {value};\n"
    formatted_output = formatted_output.rstrip(";\n")  # Remove the trailing semicolon
    argument_printout = textwrap.fill(formatted_output, width=80)  # Wrap the output text to fit in rows and columns

    # Print run configuration
    print_and_log(f"\033[1m{'-' * 80}\n{('VirtualIce Run Configuration').center(80)}\n{'-' * 80}\033[0m", logging.WARNING)

    # Display structure sets – format each structure set correctly
    structure_sets_str = ', '.join(format_structure_sets(args.structures))

    print_and_log(textwrap.fill(f"Generating {args.num_images} synthetic micrograph{'' if args.num_images == 1 else 's'} per structure set ({structure_sets_str}) using micrographs in {args.image_directory.rstrip('/')}/", width=80), logging.WARNING)
    print_and_log(f"\nInput command: {' '.join(sys.argv)}", logging.DEBUG)
    print_and_log("\nInput arguments:\n", logging.WARNING)
    print_and_log(argument_printout, logging.WARNING)
    print_and_log(f"\033[1m{'-' * 80}\033[0m\n", logging.WARNING)

    # Adjust aggregation amounts based on the total number of micrographs (each set has its own micrographs)
    args.aggregation_amount = parse_aggregation_amount(args.aggregation_amount, args.num_images * len(flattened_structures))

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
        print_and_log(f"[{pdb_id}] Downloading PDB {pdb_id}...")

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

    # Determine the largest .pdb file (which should be the symmetrized one; entries are not consistent) and remove the other one if both exist
    if os.path.exists(regular_pdb_path) and os.path.exists(symmetrized_pdb_path):
        if os.path.getsize(symmetrized_pdb_path) > os.path.getsize(regular_pdb_path):
            print_and_log(f"[{pdb_id}] Downloaded symmetrized PDB {pdb_id}")
            os.path.exists(regular_pdb_path) and os.remove(regular_pdb_path)
            os.rename(symmetrized_pdb_path, regular_pdb_path)
        else:
            print_and_log(f"[{pdb_id}] Downloaded PDB {pdb_id}")
            os.path.exists(symmetrized_pdb_path) and os.remove(symmetrized_pdb_path)
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
        print_and_log(f"Error fetching XML data: {e}", logging.ERROR)
        return None
    except request.HTTPError as e:
        print_and_log(f"HTTP Error fetching XML data: {e}", logging.ERROR)
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
            print_and_log(f"[emd_{emdb_id}] Downloading EMD-{emdb_id}...")
        request.urlretrieve(url, local_filename)

        # Decompress the downloaded file
        with gzip.open(local_filename, 'rb') as f_in:
            with open(local_filename.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the compressed file after decompression
        os.remove(local_filename)

        print_and_log(f"[emd_{emdb_id}] Download and decompression complete for EMD-{emdb_id}.")
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
        print_and_log(f"Error during scaling operation: {e}", logging.ERROR)

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

def reorient_mrc(input_mrc_path):
    """
    Reorient an MRC file so that the structure's principal axes align with the coordinate axes.

    The function performs PCA on the non-zero voxels to align the longest axis with the z-axis,
    the second longest axis with the y-axis, and the third longest axis with the x-axis.

    :param str input_mrc_path: The path to the input MRC file.
    """
    print_and_log("", logging.DEBUG)
    data = read_mrc(input_mrc_path)

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

def read_mrc(mrc_path):
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

def write_mrc(mrc_path, numpy_array, pixelsize=1.0):
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

def convert_point_to_model(point_file, output_file, circle_radius, circle_thickness, structure_name, circle_color):
    """
    Write an IMOD .mod file with particle coordinates.

    :param str point_file: Path to the input .point file.
    :param str output_file: Output file path for the .mod file.
    :param int circle_radius: Radius of the circle to be drawn at each particle location.
    :param int circle_thickness: Thickness of the circular line drawn in IMOD .mod files.
    :param str structure_name: Name of the structure to be included in the .mod file.
    :param tuple circle_color: RGB color values for the circle (0-255).

    This function writes a .mod file to the output_file path.
    """
    print_and_log("", logging.DEBUG)
    try:
        # Run point2model command with updated arguments
        color_str = f"{circle_color[0]},{circle_color[1]},{circle_color[2]}"
        output = subprocess.run([
            "point2model",
            "-circle", str(circle_radius),
            "-width", str(circle_thickness),
            "-name", structure_name,
            "-color", color_str,
            "-scat", point_file,
            output_file
        ], capture_output=True, text=True, check=True)
        print_and_log(output, logging.DEBUG)
    except subprocess.CalledProcessError:
        print_and_log("Error while converting coordinates using point2model.", logging.WARNING)
    except FileNotFoundError:
        print_and_log("point2model not found. Ensure IMOD is installed and point2model is in your system's PATH.", logging.WARNING)

def write_mod_file(coordinates, output_file, circle_radius, circle_thickness, structure_name, circle_color):
    """
    Write an IMOD .mod file with particle coordinates.

    :param list_of_tuples coordinates: List of (x, y) coordinates for the particles.
    :param str output_file: Output file path for the .mod file.
    :param int circle_radius: Radius of the circle to be drawn at each particle location.
    :param int circle_thickness: Thickness of the circular line drawn in IMOD .mod files.
    :param str structure_name: Name of the structure to be included in the .mod file.
    :param tuple circle_color: RGB color values for the circle (0-255).

    This function converts particle coordinates in a .point file to an IMOD .mod file and writes it to output_file.
    """
    print_and_log("", logging.DEBUG)
    # Write the .point file
    point_file = os.path.splitext(output_file)[0] + ".point"
    with open(point_file, 'w') as f:
        for x, y in coordinates:
            f.write(f"{x} {y} 0\n")  # Writing each coordinate as a new line in the .point file

    # Convert the .point file to a .mod file
    convert_point_to_model(point_file, output_file, circle_radius, circle_thickness, structure_name, circle_color)

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

def save_particle_coordinates(structure_name, particle_locations_with_orientations, output_path, 
                              micrograph_output_path, imod_coordinate_file, coord_coordinate_file,
                              defocus, imod_circle_radius, imod_circle_thickness, imod_circle_color):
    """
    Saves particle coordinates in specified formats (.star, .mod, .coord).

    :param str structure_name: Name of the structure.
    :param list_of_tuples particle_locations_with_orientations: List of tuples where each tuple contains:
        - tuple 1: The (x, y) coordinates of the particle.
        - tuple 2: The (alpha, beta, gamma) Euler angles representing the orientation of the particle.
    :param str output_path: Base path for the output files.
    :param bool imod_coordinate_file: Whether to output IMOD .mod files.
    :param bool coord_coordinate_file: Whether to output .coord files.
    :param float defocus: The defocus value to add to the STAR file.
    :param int imod_circle_radius: Radius of the circle drawn in IMOD .mod files.
    :param int imod_circle_thickness: Thickness of the circular line drawn in IMOD .mod files.
    :param tuple imod_circle_color: Color of the circle drawn in IMOD .mod files.

    This function writes a .star file and optionally generates .mod and .coord files.
    """
    print_and_log("", logging.DEBUG)

    # Extract particle locations and orientations
    particle_locations = [loc for loc, ori in particle_locations_with_orientations]
    orientations = [ori for loc, ori in particle_locations_with_orientations]

    # Save coordinates to .star file for the current structure
    write_all_coordinates_to_star(structure_name, micrograph_output_path + ".mrc", particle_locations, orientations, defocus)

    # Save IMOD .mod files if requested
    if imod_coordinate_file:
        # Use the color for this specific structure
        color = imod_circle_color if isinstance(imod_circle_color, tuple) else convert_color_to_rgb(imod_circle_color)
        write_mod_file(particle_locations, os.path.splitext(output_path)[0] + ".mod", imod_circle_radius, imod_circle_thickness, structure_name, color)

    # Save .coord files if requested
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
            image = read_mrc(image_path)
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
            write_mrc(binned_image_path, downsampled_image.astype(np.float32), downsample_factor * pixelsize)
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
        image_size = read_mrc(next(file for file in image_paths if file.endswith('.mrc'))).nbytes
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

def downsample_coordinate_files(structure_name, structure_set_name, binning, imod_coordinate_file, coord_coordinate_file, circle_radius, circle_thickness, circle_color):
    """
    Downsample coordinate files based on the specified binning factor.

    :param str structure_name: Name of the structure.
    :param str structure_set_name: Name of the structure set.
    :param int binning: The factor by which to downsample the coordinates.
    :param bool imod_coordinate_file: Whether to downsample and save IMOD .mod coordinate files.
    :param bool coord_coordinate_file: Whether to downsample and save .coord coordinate files.
    :param int circle_radius: Radius of the circle to be drawn at each particle location.
    :param int circle_thickness: Thickness of the circular line drawn in IMOD .mod files.
    :param tuple circle_color: RGB color values for the circle (0-255).
    """
    print_and_log("", logging.DEBUG)
    downsample_star_file(f"{structure_name}.star", f"{structure_name}_bin{binning}.star", binning)
    if imod_coordinate_file:
        for filename in os.listdir(f"{structure_set_name}/"):
            if filename.endswith(".point"):
                input_file = os.path.join(f"{structure_set_name}/", filename)
                output_point_file = os.path.join(f"{structure_set_name}/bin_{binning}/", filename.replace('.point', f'_bin{binning}.point'))
                # First downsample the .point files
                downsample_point_file(input_file, output_point_file, binning)
                # Then convert all of the .point files to .mod files
                mod_file = os.path.splitext(output_point_file)[0] + ".mod"
                convert_point_to_model(output_point_file, mod_file, circle_radius, circle_thickness, structure_name, circle_color)
    if coord_coordinate_file:
        for filename in os.listdir(f"{structure_set_name}/"):
            if filename.endswith(".coord"):
                input_file = os.path.join(f"{structure_set_name}/", filename)
                output_coord_file = os.path.join(f"{structure_set_name}/bin_{binning}/", filename.replace('.coord', f'_bin{binning}.coord'))
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
    max_num_particles_without_overlap = int(2 * input_micrograph.shape[0] * input_micrograph.shape[1] / (trimmed_mrc_array.shape[0] * trimmed_mrc_array.shape[1]) / (scale_percent/100))

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

def determine_ice_and_particle_behavior(args, structure, structure_name, micrograph, ice_scaling_fudge_factor, remaining_aggregation_amounts, context):
    """
    Determine ice and particle behaviors: First trim the volume and determine possible numbers of particles,
    then determine ice thickness, then determine particle distribution, then determine particle aggregation.

    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :param numpy_array structure: 3D numpy array of the structure for which to generate projections.
    :param str structure_name: Name of the structure.
    :param numpy_array micrograph: 2D numpy array representing the ice micrograph.
    :param float ice_scaling_fudge_factor: Fudge factor for making particles look dark enough.
    :param list remaining_aggregation_amounts: Keep track of aggregation amounts to not repeat them.
    :param str context: Context string for print statements (structure name and micrograph number).
    :return tuple: ice_thickness, ice_thickness_printout, num_particles, dist_type, non_random_dist_type, aggregation_amount_val
    """
    print_and_log("", logging.DEBUG)
    print_and_log(f"{context} Trimming the mrc ({structure_name})...")
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
        print_and_log(f"{context} Attempting to find {int(num_particles)} particle locations for {structure_name} in the micrograph...")
    else:
        print_and_log(f"{context} Choosing a random number of particles for {structure_name} to add to the micrograph...")

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

    # Project the rotated volume by summing along the z-axis
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

    # Project the rotated volume by summing along the z-axis
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

def generate_particle_locations(micrograph_image, image_size, num_particles_per_structure, half_small_image_widths, 
                                border_distance, no_edge_particles, dist_type, non_random_dist_type, 
                                aggregation_amount, allow_overlap):
    """
    Generate random/non-random locations for particles from multiple structures within an image,
    with optional aggregation in the 'micrograph' distribution type.

    Particles from multiple structures are placed without overlaps (if not allowed) across all structures.
    The size of each structure (represented by `half_small_image_width`) is taken into account when placing particles.

    :param numpy_array micrograph_image: Micrograph image (used only in the 'micrograph' distribution option).
    :param tuple image_size: The size of the image as a tuple (height, width).
    :param list num_particles_per_structure: A list containing the number of particles for each structure.
    :param list half_small_image_widths: A list containing half the widths of the particles for each structure.
    :param int border_distance: The minimum distance between particles and the image border.
    :param bool no_edge_particles: Prevent particles from being placed up to the edge of the micrograph.
    :param str dist_type: Particle location generation distribution type - 'random' or 'non_random'.
    :param str non_random_dist_type: Type of non-random distribution when dist_type is 'non_random';
                                     e.g., 'circular', 'inverse_circular', 'gaussian', or 'micrograph'.
    :param float aggregation_amount: Amount of particle aggregation.
    :param bool allow_overlap: Flag to allow overlapping particles.
    :return list_of_lists, optional_prob_map: A list of particle locations for each structure, and the optional probability map.
    """
    print_and_log("", logging.DEBUG)
    height, width = image_size

    # Initialize list for storing particle locations for each structure
    all_particle_locations = [[] for _ in range(len(num_particles_per_structure))]

    # Adjust border distance for each structure
    if no_edge_particles:
        border_distances = [max(border_distance, hsw) for hsw in half_small_image_widths]
    else:
        border_distances = [0 for _ in half_small_image_widths]  # Minimum border distance is 0

    # Compute the maximum half_small_image_width for overlap checking
    max_half_small_image_width = max(half_small_image_widths)

    # Step 1: Calculate the total number of particles across all structures
    total_particles = sum(num_particles_per_structure)

    # Step 2: Generate candidate positions for all particles
    # Depending on the distribution, generate positions accordingly
    # We will generate extra candidates to account for overlap rejections
    candidate_multiplier = 1.5 if not allow_overlap else 1.0  # Generate more candidates if overlaps are not allowed
    num_candidates = int(total_particles * candidate_multiplier)

    # Initialize inverse_normalized_prob_map to None
    inverse_normalized_prob_map = None

    # Generate positions based on the distribution type
    if dist_type == 'random':
        x_coords = np.random.randint(0, width, size=num_candidates)
        y_coords = np.random.randint(0, height, size=num_candidates)
        candidate_positions = np.column_stack((x_coords, y_coords))
    elif dist_type == 'non_random':
        if non_random_dist_type == 'circular':
            # Generate positions in a circular cluster
            cluster_center = (width // 2, height // 2)
            radius = min(cluster_center[0], cluster_center[1])
            angles = np.random.uniform(0, 2 * np.pi, num_candidates)
            radii = np.sqrt(np.random.uniform(0, 1, num_candidates)) * radius
            x_coords = cluster_center[0] + radii * np.cos(angles)
            y_coords = cluster_center[1] + radii * np.sin(angles)
            candidate_positions = np.column_stack((x_coords.astype(int), y_coords.astype(int)))
        elif non_random_dist_type == 'inverse_circular':
            # Exclude a circular region in the center
            exclusion_center = (width // 2, height // 2)
            exclusion_radius = min(width, height) * 0.3  # Adjust as needed

            # Compute max_radius as the maximum distance from the exclusion_center to the image corners
            corners = [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]
            distances = [np.hypot(exclusion_center[0] - x, exclusion_center[1] - y) for x, y in corners]
            max_radius = max(distances)

            # Generate positions in an annular region between exclusion_radius and max_radius
            angles = np.random.uniform(0, 2 * np.pi, num_candidates)
            radii = np.sqrt(np.random.uniform(exclusion_radius**2, max_radius**2, num_candidates))
            x_coords = exclusion_center[0] + radii * np.cos(angles)
            y_coords = exclusion_center[1] + radii * np.sin(angles)
            candidate_positions = np.column_stack((x_coords.astype(int), y_coords.astype(int)))
        elif non_random_dist_type == 'gaussian':
            # Generate positions from multiple Gaussian clusters
            num_gaussians = np.random.randint(1, 6)
            particles_per_gaussian = num_candidates // num_gaussians
            candidate_positions = []
            for _ in range(num_gaussians):
                center_x = np.random.uniform(0, width)
                center_y = np.random.uniform(0, height)
                stddev = min(width, height) / 10  # Adjust as needed
                x_coords = np.random.normal(center_x, stddev, particles_per_gaussian)
                y_coords = np.random.normal(center_y, stddev, particles_per_gaussian)
                candidate_positions.extend(zip(x_coords.astype(int), y_coords.astype(int)))
            candidate_positions = np.array(candidate_positions)
        elif non_random_dist_type == 'micrograph':
            # Use the micrograph to generate a probability map
            sigma = min(width, height) / 1  # Need to divide by something otherwise it breaks
            itk_image = sitk.GetImageFromArray(micrograph_image)
            filtered_micrograph_itk = sitk.SmoothingRecursiveGaussian(itk_image, sigma=sigma)
            filtered_micrograph = sitk.GetArrayFromImage(filtered_micrograph_itk)

            # Invert the filtered image to assign higher probability to lower pixel values
            inverted_micrograph = np.max(filtered_micrograph) - filtered_micrograph

            # Normalize the inverted image to get probabilities
            prob_map = inverted_micrograph / np.sum(inverted_micrograph)

            # Flatten the probability map
            flat_prob_map = prob_map.ravel()

            # Generate candidate positions based on the probability map
            indices = np.random.choice(len(flat_prob_map), size=num_candidates, p=flat_prob_map)
            y_coords, x_coords = np.unravel_index(indices, prob_map.shape)
            candidate_positions = np.column_stack((x_coords, y_coords))

            # Apply aggregation if aggregation_amount > 0
            if aggregation_amount > 0:
                # Initialize clump centers
                num_clumps = max(1, int(total_particles * np.random.uniform(0.1, 0.5)))
                clump_centers = np.array([
                    (
                        np.random.randint(0, width),
                        np.random.randint(0, height)
                    ) for _ in range(num_clumps)
                ])

                # Adjust candidate positions closer to clump centers
                aggregation_factor = aggregation_amount / 11.0

                for idx in range(len(candidate_positions)):
                    x, y = candidate_positions[idx]
                    clump_center = clump_centers[np.random.choice(len(clump_centers))]
                    shift_x = int((clump_center[0] - x) * aggregation_factor)
                    shift_y = int((clump_center[1] - y) * aggregation_factor)
                    # To make it so clumps aren't universal attractors, only update the particle location if
                    # aggregation_factor is less than a random number (i.e., use this as a probability of changing)
                    if np.random.random() <= aggregation_factor:
                        candidate_positions[idx, 0] = x + shift_x
                        candidate_positions[idx, 1] = y + shift_y

            # Generate the inverse normalized probability map for output
            inverse_normalized_prob_map = 0.8 + 0.4 * (1 - (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min()))
        else:
            raise ValueError(f"Unknown non-random distribution type: {non_random_dist_type}")
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    # Step 3: Filter candidate positions that are outside the valid area (considering borders)
    valid_candidates = []
    for idx, (x, y) in enumerate(candidate_positions):
        # Check if the position is within the image bounds
        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        # Use the minimum border distance across structures to ensure positions are valid
        min_border_dist = min(border_distances)
        if x < min_border_dist or x >= width - min_border_dist or y < min_border_dist or y >= height - min_border_dist:
            continue

        valid_candidates.append((x, y))

    if not valid_candidates:
        print_and_log("Warning: No valid candidate positions found for particles.", logging.WARNING)
        # Return empty lists
        if dist_type == 'non_random' and non_random_dist_type == 'micrograph':
            return [[] for _ in num_particles_per_structure], inverse_normalized_prob_map
        else:
            return [[] for _ in num_particles_per_structure], None

    valid_candidates = np.array(valid_candidates)

    # Step 4: Assign candidate positions to structures
    particle_counts = [0] * len(num_particles_per_structure)
    total_required_particles = sum(num_particles_per_structure)
    candidate_idx = 0

    if allow_overlap:
        # Overlaps are allowed between all particles
        # Assign positions without overlap checks
        structure_indices = list(range(len(num_particles_per_structure)))
        while candidate_idx < len(valid_candidates) and sum(particle_counts) < total_required_particles:
            for s_idx in structure_indices:
                if particle_counts[s_idx] < num_particles_per_structure[s_idx]:
                    x, y = valid_candidates[candidate_idx]
                    # Check if the position is within the required border distance for this structure
                    border_dist = border_distances[s_idx]
                    if x >= border_dist and x < width - border_dist and y >= border_dist and y < height - border_dist:
                        all_particle_locations[s_idx].append((x, y))
                        particle_counts[s_idx] += 1
                    candidate_idx += 1
                    if candidate_idx >= len(valid_candidates):
                        break
            if candidate_idx >= len(valid_candidates):
                break
    else:
        # Overlaps are not allowed within the same structure
        # Overlaps between particles of different structures are allowed
        # Initialize per-structure placed positions
        placed_positions_list = [[] for _ in range(len(num_particles_per_structure))]

        structure_indices = list(range(len(num_particles_per_structure)))
        while candidate_idx < len(valid_candidates) and sum(particle_counts) < total_required_particles:
            x, y = valid_candidates[candidate_idx]
            candidate_assigned = False  # Flag to determine if candidate was assigned

            # Try to assign the candidate to structures in a round-robin fashion
            for s_idx in structure_indices:
                if particle_counts[s_idx] < num_particles_per_structure[s_idx]:
                    # Check if the position is within the required border distance for this structure
                    border_dist = border_distances[s_idx]
                    if x < border_dist or x >= width - border_dist or y < border_dist or y >= height - border_dist:
                        continue  # Invalid position for this structure

                    # Check overlap with existing particles of the same structure
                    if placed_positions_list[s_idx]:
                        tree = KDTree(placed_positions_list[s_idx])
                        distances, _ = tree.query([(x, y)], k=1)
                        min_distance = half_small_image_widths[s_idx] * 0.9  # Adjust the factor as needed
                        if distances[0] < min_distance:
                            continue  # Overlaps with existing particle of the same structure

                    # Proceed to assign the particle to the structure
                    all_particle_locations[s_idx].append((x, y))
                    particle_counts[s_idx] += 1
                    placed_positions_list[s_idx].append((x, y))
                    candidate_assigned = True
                    break  # Candidate assigned to a structure
            candidate_idx += 1

    if dist_type == 'non_random' and non_random_dist_type == 'micrograph':
        return all_particle_locations, inverse_normalized_prob_map
    else:
        return all_particle_locations, None

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

def filter_out_overlapping_particles(particle_locations, half_small_image_width, allowed_overlap):
    """
    Filter out overlapping particles based on the center-to-center distance.

    :param list_of_tuples particle_locations: List of (x, y) coordinates of particle locations.
    :param int half_small_image_width: Half the width of a small image.
    :param float allowed_overlap: Fudge factor for determining how close non-overlapping particles can be.
    :return list_of_tuples: List of (x, y) coordinates of non-overlapping particle locations.
    """
    print_and_log("", logging.DEBUG)
    if not particle_locations:
        return []

    # Build a KDTree for the particle locations
    tree = KDTree(particle_locations)

    # Define the minimum distance required to avoid overlap
    min_distance = allowed_overlap * half_small_image_width

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
    Blend small images (particles) from multiple structures into a large image (micrograph).
    Also performs junk filtering, edge filtering, and generates coordinate files.

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
    all_small_images = input_options['small_images']  # A list of small images for all structures
    all_particle_locations = input_options['particle_locations']  # A list of particle locations for all structures
    all_orientations = input_options['orientations']  # A list of orientations for all structures
    structure_names = input_options['structure_names']  # A list of structure names
    output_paths = output_options['output_paths']  # Output paths for each structure

    # Extract junk labels options
    no_junk_filter = junk_labels_options['no_junk_filter']
    flip_x = junk_labels_options['flip_x']
    flip_y = junk_labels_options['flip_y']
    json_scale = junk_labels_options['json_scale']
    polygon_expansion_distance = junk_labels_options['polygon_expansion_distance']

    # Misc options
    allowed_overlap = particle_and_micrograph_generation_options['allowed_overlap']

    # Initialize filtered particle locations for all structures
    filtered_all_particle_locations = []

    # Step 1: Junk Filtering
    json_file_path = os.path.splitext(input_options['large_image_path'])[0] + ".json"
    for i, structure_name in enumerate(structure_names):
        particle_locations = all_particle_locations[i]

        if not no_junk_filter:
            if os.path.exists(json_file_path):
                polygons = read_polygons_from_json(json_file_path, polygon_expansion_distance, flip_x, flip_y, expand=True)
                filtered_particle_locations = filter_coordinates_outside_polygons(particle_locations, json_scale, polygons)
                num_particles_removed = len(particle_locations) - len(filtered_particle_locations)
                print_and_log(f"{context} {num_particles_removed} obscured particle{'' if num_particles_removed == 1 else 's'} removed for {structure_name} based on the JSON file.")
            else:
                print_and_log(f"{context} No JSON file found for bad micrograph areas: {json_file_path}", logging.WARNING)
                filtered_particle_locations = particle_locations
        else:
            print_and_log(f"{context} Skipping junk filtering for {structure_name} (i.e., not using JSON file)")
            filtered_particle_locations = particle_locations

        filtered_all_particle_locations.append(filtered_particle_locations)

    # Step 2: Edge Particle Filtering
    final_particle_locations = []
    for i, filtered_particle_locations in enumerate(filtered_all_particle_locations):
        remaining_particle_locations = filtered_particle_locations[:]
        structure_name = structure_names[i]
        half_small_image_width = input_options['half_small_image_widths'][i]  # Handle different sizes for each structure

        if not particle_and_micrograph_generation_options['save_edge_coordinates']:
            for x, y in filtered_particle_locations:
                reduced_sidelength = int(np.ceil(half_small_image_width * 100 / (100 + particle_and_micrograph_generation_options['scale_percent'])))
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
                print_and_log(f"{context} {num_particles_removed} edge particle{'' if num_particles_removed == 1 else 's'} removed for {structure_name}.")
            else:
                print_and_log(f"{context} 0 edge particles removed for {structure_name}.")
        final_particle_locations.append(remaining_particle_locations)

    # Step 3: Overlapping Particle Filtering (for Coordinate Files)
    final_filtered_particle_locations = []
    for i, particle_locations in enumerate(final_particle_locations):
        if not particle_and_micrograph_generation_options['save_overlapping_coords']:
            half_small_image_width = input_options['half_small_image_widths'][i]
            filtered_particle_locations = filter_out_overlapping_particles(particle_locations, half_small_image_width, allowed_overlap)
            num_particles_removed = len(particle_locations) - len(filtered_particle_locations)
            if num_particles_removed > 0:
                print_and_log(f"{context} {num_particles_removed} overlapping particle{'' if num_particles_removed == 1 else 's'} removed for {structure_names[i]}.")
            else:
                print_and_log(f"{context} 0 overlapping particles removed for {structure_names[i]}.")
        else:
            filtered_particle_locations = particle_locations

        final_filtered_particle_locations.append(filtered_particle_locations)

    # Step 4: Ensure that small_images and particle_locations match in length
    for i in range(len(all_small_images)):
        if len(all_small_images[i]) > len(all_particle_locations[i]):
            all_small_images[i] = all_small_images[i][:len(all_particle_locations[i])]

    # Step 5: Normalize the input micrograph to itself
    large_image[:, :] = (large_image[:, :] - large_image[:, :].mean()) / large_image[:, :].std()

    # Step 6: Create the collage of particles on the micrograph for all structures
    for i in range(len(all_small_images)):
        collage = create_collage(large_image, all_small_images[i], all_particle_locations[i], 
                                 particle_and_micrograph_generation_options['gaussian_variance'])

        # If a probability map is provided, adjust the collage based on local ice thickness
        if 'prob_map' in input_options and input_options['prob_map'] is not None:
            collage *= simulation_options['scale'] * input_options['prob_map']
        else:
            collage *= simulation_options['scale']

        # Blend the collage with the large image
        large_image = large_image + collage

    # Step 7: Normalize the resulting micrograph to itself
    blended_image = (large_image - large_image.mean()) / large_image.std()

    # Step 8: Combine filtered_particle_locations with orientations for easier passing
    filtered_particle_locations_with_orientations = []
    for i, structure_name in enumerate(structure_names):
        for loc, ori in zip(final_filtered_particle_locations[i], all_orientations[i]):
            filtered_particle_locations_with_orientations.append((loc, ori))

    # Initialize a list to store the number of saved coordinates for each structure
    num_particles_saved_per_structure = []

    # Step 9: Save particle coordinates to coordinate files (.star, .mod, .coord) for each structure
    for i, structure_name in enumerate(structure_names):
        structure_particle_locations_with_orientations = [(loc, ori) for loc, ori in zip(final_filtered_particle_locations[i], all_orientations[i])]
        save_particle_coordinates(structure_name, structure_particle_locations_with_orientations, output_paths[i], output_options['output_file_path'],
                                  output_options['imod_coordinate_file'], output_options['coord_coordinate_file'], defocus,
                                  output_options['imod_circle_radius'], output_options['imod_circle_thickness'], output_options['imod_circle_color'][i])
        # Track the number of saved particles for this structure
        num_particles_saved_per_structure.append(len(structure_particle_locations_with_orientations))

    return blended_image, filtered_particle_locations_with_orientations, num_particles_saved_per_structure

def add_images(input_options, particle_and_micrograph_generation_options, simulation_options, 
               junk_labels_options, output_options, context, defocus):
    """
    Add small images or particles to a large image and save the resulting micrograph.

    This version handles multiple structures per micrograph, combining their projections and particle
    locations into a single micrograph.

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
    all_small_images = input_options['small_images']
    all_particle_locations = input_options['particle_locations']
    all_orientations = input_options['orientations']
    pixelsize = input_options['pixelsize']
    structure_names = input_options['structure_names']
    output_paths = output_options['output_paths']

    # Ensure that we have enough small images and particle locations for all structures
    total_particles = sum([len(p) for p in all_small_images])

    # Initialize the probability map (if micrograph distribution is used)
    prob_map = input_options.get('prob_map', None)

    # Proceed with blending images
    particle_count = sum([len(locations) for locations in all_particle_locations])

    # Concatenate all structure names to form the suffix for the output file basename
    structure_names_combined = "_".join(structure_names)

    # Get the directory and base filename from the first output path
    output_dir = os.path.dirname(output_paths[0])
    base_filename = os.path.splitext(os.path.basename(large_image_path))[0]

    # Build the full path using the directory, base filename, and combined structure names
    output_file_path = os.path.join(output_dir, f"{base_filename}_{structure_names_combined}")
    output_options['output_file_path'] = output_file_path

    # Blend the images and filter coordinates, then save the output
    result_image, filtered_particle_locations, num_particles_saved_per_structure = blend_images(
        input_options, particle_and_micrograph_generation_options,
        simulation_options, junk_labels_options, output_options, context, defocus
    )

    # Save the resulting micrograph in the specified formats
    if output_options['save_as_mrc']:
        print_and_log(f"{context} Writing synthetic micrograph: {output_file_path}.mrc...")
        write_mrc(f"{output_file_path}.mrc", (result_image - np.mean(result_image)) / np.std(result_image), pixelsize)  # Write normalized mrc (mean = 0, std = 1)

    if output_options['save_as_png']:
        # Normalize from 0 to 255 and flip the image before saving
        result_image -= result_image.min()
        result_image = result_image / result_image.max() * 255.0
        print_and_log(f"{context} Writing synthetic micrograph: {output_file_path}.png...")
        cv2.imwrite(f"{output_file_path}.png", np.flip(result_image, axis=0))

    if output_options['save_as_jpeg']:
        # Normalize from 0 to 255 and flip the image before saving
        result_image -= result_image.min()
        result_image = result_image / result_image.max() * 255.0
        print_and_log(f"{context} Writing synthetic micrograph: {output_file_path}.jpeg...")
        cv2.imwrite(f"{output_file_path}.jpeg", np.flip(result_image, axis=0), [cv2.IMWRITE_JPEG_QUALITY, output_options['jpeg_quality']])

    # Return total_particles and the list of num_particles_saved_per_structure
    return total_particles, num_particles_saved_per_structure

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
                write_mrc(particle_path, cp.asnumpy(cropped_particle).astype(np.float32), pixelsize)
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
                    write_mrc(particle_path, cp.asnumpy(cropped_particle).astype(np.float32), pixelsize)
                    cropped_count += 1
                total_cropped += cropped_count

    return total_cropped

def crop_particles(micrograph_path, particle_rows, particles_dir, box_size, pixelsize, max_crop_particles, context):
    """
    Crops particles from a single micrograph.

    :param str micrograph_path: Path to the micrograph.
    :param DataFrame particle_rows: DataFrame rows of particles to be cropped from the micrograph.
    :param str particles_dir: Directory to save cropped particles.
    :param int box_size: The box size in pixels for the cropped particles.
    :param float pixelsize: Pixel size of the micrograph.
    :param int max_crop_particles: The maximum number of particles to crop from the micrograph.
    :param str context: Context string for print statements.
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
            write_mrc(particle_path, cropped_particle.astype(np.float32), pixelsize)
            cropped_count += 1

    print_and_log(f"{context} Cropped {cropped_count} particles from {micrograph_path}")
    return cropped_count

def crop_particles_from_micrographs(structure_name, structure_set_name, box_size, pixelsize, max_crop_particles, num_cpus, use_gpu, gpu_ids):
    """
    Crops particles from micrographs based on coordinates specified in a micrograph STAR file
    and saves them with a specified box size. This function operates in parallel, using multiple GPUs
    or multiple CPU cores to process different micrographs concurrently.

    :param str structure_name: The structure name.
    :param str structure_set_name: The directory containing the current structure set.
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
      convention of '{structure_name}.star' or {structure_set_name}/{structure_name}.star containing the
      necessary coordinates for cropping.
    """
    print_and_log("", logging.DEBUG)
    structure_name = structure_name if structure_name.isalnum() else structure_name.split('/')[-1].split('\\')[-1].rsplit('.', 1)[0]  # Strips file extension if the user inputted a local file
    structure_set_number = int(structure_set_name.split('_')[-1])
    context = f"[SS #{structure_set_number} | {structure_name}]"
    print_and_log(f"{context} Cropping particles...")
    particles_dir = os.path.join(structure_set_name, f'Particles_{structure_name}/')
    os.makedirs(particles_dir, exist_ok=True)

    star_file_path = next((path for path in [f'{structure_set_name}/{structure_name}.star', f'{structure_name}.star'] if os.path.isfile(path)), None)  # Hack to get around a bug where the star file is either in the base directory or structure directory.
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
                            print_and_log(f"{context} Micrograph not found: {micrograph_path}", logging.WARNING)
                            continue
                        print_and_log(f"{context} Extracting {len(particle_rows)} {structure_name} particles for {micrograph_name}...", logging.DEBUG)
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
                    print_and_log(f"{context} Micrograph not found: {micrograph_path}", logging.WARNING)
                    continue

                print_and_log(f"{context} Extracting {len(particle_rows)} {structure_name} particles for {micrograph_name}...", logging.DEBUG)
                future = executor.submit(crop_particles, micrograph_path, particle_rows, particles_dir, box_size, pixelsize, max_crop_particles, context)
                futures.append(future)

            for future in futures:
                total_cropped += future.result()

    return total_cropped

def process_single_micrograph(args, structures, line, total_structures, structure_index,
                              micrograph_usage_count, remaining_aggregation_amounts, micrograph_number, structure_set_name):
    """
    Process a single micrograph for a set of structures.

    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :param list structures: List of tuples, each containing (structure_name, structure, mass, ice_scaling_fudge_factor).
    :param str line: The line containing the micrograph name and defocus value.
    :param int total_structures: Total number of structure sets requested.
    :param int structure_index: Index of the current structure set.
    :param dict micrograph_usage_count: Dictionary to keep track of the usage count of each unique micrograph.
    :param list remaining_aggregation_amounts: List to track remaining aggregation amounts.
    :param int micrograph_number: Current micrograph number (1-indexed).
    :param str structure_set_name: The name of the directory for this structure set.
    :return tuple: Number of particles projected, number of particles with saved coordinates.
    """
    print_and_log("", logging.DEBUG)

    # Reseed the random number generators per process to ensure unique random numbers
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    np.random.seed(seed)
    random.seed(seed)

    # Parse the 'micrograph_name.mrc defocus' line
    fname, defocus = line.strip().split()[:2]
    fname = os.path.splitext(os.path.basename(fname))[0]
    micrograph = read_mrc(f"{args.image_directory}/{fname}.mrc")
    mean, gaussian_variance = estimate_noise_parameters(micrograph)

    # Track each micrograph's usage count for naming purposes
    micrograph_usage_count[fname] = micrograph_usage_count.get(fname, 0) + 1
    # Generate the repeat number suffix for the filename
    repeat_suffix = f"_{micrograph_usage_count[fname]}" if micrograph_usage_count[fname] > 1 else ""

    # Add context for printout
    context = f"[SS #{structure_index + 1} | {micrograph_number}/{args.num_images}]"
    num_hyphens = '-' * (57 + len(f"{structure_index + 1}{total_structures}{fname}"))
    print_and_log(f"\n\033[1m{num_hyphens}\033[0m", logging.WARNING)
    print_and_log(f"Generating synthetic micrograph #{micrograph_number} for SS #{structure_index + 1} ({structure_index + 1}/{total_structures}) from {fname}...", logging.WARNING)
    print_and_log(f"\033[1m{num_hyphens}\033[0m\n", logging.WARNING)

    # Determine if overlap is allowed for this micrograph
    if args.allow_overlap_random:
        args.allow_overlap = bool(random.getrandbits(1))
    print_and_log(f"{context} {'Allowing' if args.allow_overlap else 'Not allowing'} overlapping particles for this micrograph.")

    # Initialize aggregation amounts for this micrograph
    remaining_aggregation_amounts = list(remaining_aggregation_amounts)

    # Initialize totals for this micrograph
    total_num_particles_projected = 0
    total_num_particles_with_saved_coordinates = 0

    # Initialize lists to store combined particles, locations, and orientations for all structures
    all_particle_locations = []
    all_orientations = []
    all_particles = []  # Will hold the processed particle stacks for each structure
    half_small_image_widths = []
    num_particles_per_structure = []

    # Step 1: Loop over each structure to determine maximum number of particles and other relevant parameters
    for structure_name, structure, mass, ice_scaling_fudge_factor in structures:
        # Determine ice and particle behavior parameters for each structure
        ice_thickness, ice_thickness_printout, num_particles, dist_type, non_random_dist_type, aggregation_amount_val = determine_ice_and_particle_behavior(
            args, structure, structure_name, micrograph, ice_scaling_fudge_factor, remaining_aggregation_amounts, context
        )
        num_particles = num_particles // len(structures)  # Evenly divide the total number of particles amongst the structures in the set
        print_and_log(f"{context} {num_particles} particles of {structure_name} will be added to the micrograph")

        # Store the number of particles for this structure
        num_particles_per_structure.append(num_particles)
        # Store half the width of the small image for each structure (used in generating locations)
        half_small_image_widths.append(structure.shape[0] // 2)

    # Step 2: Generate particle locations for all structures using round-robin placement
    particle_locations, prob_map = generate_particle_locations(
        micrograph_image=micrograph,
        image_size=micrograph.shape,
        num_particles_per_structure=num_particles_per_structure,
        half_small_image_widths=half_small_image_widths,
        border_distance=args.border,
        no_edge_particles=args.no_edge_particles,
        dist_type=dist_type,
        non_random_dist_type=non_random_dist_type,
        aggregation_amount=aggregation_amount_val,
        allow_overlap=args.allow_overlap
    )

    # Step 3: Loop over each structure again to generate projections, add noise, and apply CTF
    for idx, (structure_name, structure, mass, ice_scaling_fudge_factor) in enumerate(structures):
        print_and_log(f"{context} Projecting the structure volume ({structure_name}) {num_particles_per_structure[idx]} times...")

        # Generate projections with the specified orientation mode for the current structure
        particles, orientations = generate_projections(
            structure, 
            num_particles_per_structure[idx], 
            args.orientation_mode, 
            args.preferred_angles,
            args.angle_variation, 
            args.preferred_weight, 
            args.cpus, 
            args.use_gpu, 
            args.gpu_ids
        )

        # Store projections, orientations, and particle locations for this structure
        all_orientations.append(orientations)
        all_particle_locations.append(particle_locations[idx])  # Keep particle locations separated for each structure

        # Step 4: Simulate noise, damage, and apply CTF to this structure's particles
        print_and_log(f"{context} Simulating pixel-level Poisson noise{f' and dose damage' if args.dose_damage != 'None' else ''} across {args.num_simulated_particle_frames} particle frame{'s' if args.num_simulated_particle_frames != 1 else ''} for {structure_name}...")
        if args.use_gpu:
            noisy_particles = add_poisson_noise_gpu(particles, args.num_simulated_particle_frames, args.dose_a, args.dose_b,
                                                    args.dose_c, args.apix, args.gpu_ids)
        else:
            noisy_particles = add_poisson_noise(particles, args.num_simulated_particle_frames, args.dose_a, args.dose_b,
                                                args.dose_c, args.apix, args.cpus)

        # Apply CTF to the noisy particles
        print_and_log(f"{context} Applying CTF to {structure_name} based on defocus ({float(defocus):.4f} microns) and microscope parameters ({args.voltage} keV, AmpCont: {args.ampcont}%, Cs: {args.Cs} mm, Pixelsize: {args.apix} Angstroms) of the ice micrograph...")
        noisy_particles_CTF = apply_ctfs_with_eman2(noisy_particles, [defocus] * len(noisy_particles), args.ampcont, args.bfactor,
                                                    args.apix, args.Cs, args.voltage, args.cpus)

        # Store the noisy particles for this structure
        all_particles.append(noisy_particles_CTF)

    structure_results = []

    # Step 5: Blend all particles into the micrograph and save coordinates
    print_and_log(f"{context} Adding {len(all_particles) * num_particles} particles to the micrograph{f' {dist_type}ly ({non_random_dist_type})' if dist_type == 'non_random' else f' {dist_type}ly' if dist_type else ''}{f' with aggregation amount of {aggregation_amount_val:.1f}' if args.distribution in ('m','micrograph') else ''} while adding Gaussian (white) noise and simulating a average relative ice thickness of {ice_thickness_printout:.1f} nm...")
    input_options = {
        'large_image_path': f"{args.image_directory}/{fname}.mrc",
        'large_image': micrograph,
        'small_images': all_particles,  # Pass the processed noisy particles for all structures
        'pixelsize': args.apix,
        'structure_names': [structure[0] for structure in structures],  # Names of the structures
        'orientations': all_orientations,
        'particle_locations': all_particle_locations,
        'half_small_image_widths': half_small_image_widths
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
        'save_overlapping_coords': args.save_overlapping_coords,
        'allowed_overlap': args.allowed_overlap
    }
    simulation_options = {'scale': ice_thickness}
    junk_labels_options = {
        'no_junk_filter': args.no_junk_filter,
        'flip_x': args.flip_x,
        'flip_y': args.flip_y,
        'json_scale': args.json_scale,
        'polygon_expansion_distance': args.polygon_expansion_distance
    }

    # Use the structure_set_name for output paths to ensure consistency with generate_micrographs
    output_options = {
        'save_as_mrc': args.mrc,
        'save_as_png': args.png,
        'save_as_jpeg': args.jpeg,
        'jpeg_quality': args.jpeg_quality,
        'imod_coordinate_file': args.imod_coordinate_file,
        'coord_coordinate_file': args.coord_coordinate_file,
        'output_paths': [f"{structure_set_name}/{fname}_{structure[0]}{repeat_suffix}" for structure in structures],
        'imod_circle_radius': args.imod_circle_radius,
        'imod_circle_thickness': args.imod_circle_thickness,
        'imod_circle_color': [convert_color_to_rgb('random') if args.imod_circle_color == 'random' else args.imod_circle_color for _ in structures]
    }

    num_particles_projected, num_particles_saved_per_structure = add_images(
        input_options, particle_and_micrograph_generation_options,
        simulation_options, junk_labels_options, output_options, context, defocus
    )

    # Distribute the particles among the structures
    num_structures = len(structures)
    particles_per_structure = num_particles_projected // num_structures

    structure_results = []
    for idx, structure in enumerate(structures):
        structure_name = structure[0]
        # Assign any remaining particles to the last structure
        if idx == num_structures - 1:
            structure_projected = num_particles_projected - (particles_per_structure * (num_structures - 1))
        else:
            structure_projected = particles_per_structure
        # Use the actual number of saved particles for this structure from `num_particles_saved_per_structure`
        structure_saved = num_particles_saved_per_structure[idx]
        structure_results.append((structure_name, structure_projected, structure_saved))

    return structure_results

def process_single_structure(sub_structure_input, args):
    """
    Process a single structure: download, convert, and estimate mass.

    :param str sub_structure_input: The input structure (PDB ID, MRC file, etc.).
    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :return tuple: The structure name, structure data, mass, and ice scaling fudge factor.
    """
    result = process_structure_input(sub_structure_input, args.max_emdb_size, args.std_threshold, args.apix)
    if result:
        structure_name, structure_type = result
        if structure_type == "pdb":
            mass = convert_pdb_to_mrc(structure_name, args.apix, args.pdb_to_mrc_resolution)
            structure = reorient_mrc(f'{structure_name}.mrc')
            print_and_log(f"[{structure_name}] Reoriented MRC")
            print_and_log(f"[{structure_name}] Mass of PDB: {mass} kDa")
            ice_scaling_fudge_factor = 4.8  # Larger number = darker particles
        elif structure_type == "mrc":
            mass = int(estimate_mass_from_map(structure_name))
            if args.reorient_mrc:
                structure = reorient_mrc(f'{structure_name}.mrc')  # Reorient if requested
                print_and_log(f"[{structure_name}] Reoriented MRC")
            else:
                structure = read_mrc(f'{structure_name}.mrc')
            print_and_log(f"[{structure_name}] Estimated mass of MRC: {mass} kDa")
            ice_scaling_fudge_factor = 2.9

        return structure_name, structure, mass, ice_scaling_fudge_factor
    return None

def generate_micrographs(args, structure_set, structure_set_index, total_structure_sets):
    """
    Generate synthetic micrographs for a specified set of structures.

    This function orchestrates the workflow for generating synthetic micrographs
    for a set of structures. It performs file download, conversion, image shuffling,
    and iterates through the generation process for each selected image. It also
    handles cleanup operations post-generation.

    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :param list structure_set: List of structures to be projected onto the micrographs.
    :param int structure_set_index: The index of the current structure set.
    :param int total_structure_sets: The total number of structure sets requested.

    :return int, int, list: Total number of particles projected, total number of particles saved to coordinate files, and list of box sizes for each structure.
    """
    print_and_log("", logging.DEBUG)

    # Create a new directory for this structure set
    structure_set_name = f"structure_set_{structure_set_index + 1}"
    context = f"[SS #{structure_set_index + 1}]"
    os.makedirs(structure_set_name, exist_ok=True)

    total_num_particles_projected = 0
    total_num_particles_with_saved_coordinates = 0
    micrograph_usage_count = {}  # Dictionary to keep track of repeating micrograph names if the image list was extended
    remaining_aggregation_amounts = list()

    # Store the box sizes for each structure
    box_sizes = []

    # Process and download structures, convert PDBs, and estimate masses
    # Iterate over the structure_set, which may contain single structures or multiple structures
    structures = []
    tasks = []  # Parallel processing tasks
    with ProcessPoolExecutor(max_workers=args.cpus) as executor:
        for structure_input in structure_set:
            if isinstance(structure_input, list):
                # Handle multiple structures per micrograph (list of lists)
                for sub_structure_input in structure_input:
                    tasks.append(executor.submit(process_single_structure, sub_structure_input, args))
            else:
                # Handle single structure per micrograph
                tasks.append(executor.submit(process_single_structure, structure_input, args))

        # Collect the results from parallel tasks
        for future in tasks:
            result = future.result()
            if result:
                structure_name, structure, mass, ice_scaling_fudge_factor = result
                structures.append((structure_name, structure, mass, ice_scaling_fudge_factor))
                box_sizes.append(structure.shape[0])  # Store box size for this structure

    # Write STAR headers for each structure
    for structure_name, _, _, _ in structures:
        write_star_header(structure_name, args.apix, args.voltage, args.Cs)

    def format_structure_list(structure_set):
        """
        Format the structure set into a flattened list of structure names for printing.

        :param list structure_set: A list of structures, which can contain single or multiple structures.
        :return list: A list of structure names as strings for printing.
        """
        formatted_list = []
        # Iterate over the structure_set, which may contain single or multiple structures
        for structure_input in structure_set:
            if isinstance(structure_input, list):  # Handle multiple structures per micrograph (list of lists)
                formatted_list.extend([str(sub_structure) for sub_structure in structure_input])
            else:  # Handle single structure per micrograph
                formatted_list.append(str(structure_input))
        return formatted_list

    # Shuffle and possibly extend the ice images
    formatted_structure_list = format_structure_list(structure_set)
    print_and_log(f"{context} Selecting {args.num_images} random ice micrographs for Structure Set (SS): {', '.join(formatted_structure_list)}...")
    selected_images = extend_and_shuffle_image_list(args.num_images, args.image_list_file)

    # Initialize dictionaries to store particle counts for each structure
    structure_particles_projected = {structure[0]: 0 for structure in structures}
    structure_particles_saved = {structure[0]: 0 for structure in structures}

    # Main loop for generating micrographs
    if args.parallelize_micrographs > 1:
        # Parallel micrograph generation using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.parallelize_micrographs) as executor:
            futures = []
            for i, line in enumerate(selected_images):
                micrograph_number = i + 1  # Micrograph number (1-indexed)
                # Submit each micrograph generation task
                future = executor.submit(
                    process_single_micrograph, args, structures, line, total_structure_sets,
                    structure_set_index, micrograph_usage_count, remaining_aggregation_amounts, 
                    micrograph_number, structure_set_name
                )
                futures.append(future)

            # Collect results from parallel tasks
            for future in futures:
                # Aggregate the results from each task
                results = future.result()
                for structure_name, projected, saved in results:
                    structure_particles_projected[structure_name] += projected
                    structure_particles_saved[structure_name] += saved
    else:
        # Sequential micrograph generation
        for i, line in enumerate(selected_images):
            micrograph_number = i + 1  # Micrograph number (1-indexed)
            results = process_single_micrograph(
                args, structures, line, total_structure_sets, structure_set_index, 
                micrograph_usage_count, remaining_aggregation_amounts, micrograph_number, 
                structure_set_name
            )
            for structure_name, projected, saved in results:
                structure_particles_projected[structure_name] += projected
                structure_particles_saved[structure_name] += saved

    # Downsample micrographs and coordinate files if requested
    if args.binning > 1:
        print_and_log(f"{context} Binning/Downsampling micrographs by {args.binning} by Fourier cropping...")
        parallel_downsample_micrographs(f"{structure_set_name}/", args.binning, args.apix, args.cpus, args.use_gpu, args.gpu_ids)
        for structure_name, _, _, _ in structures:
            downsample_coordinate_files(structure_name, structure_set_name, args.binning, args.imod_coordinate_file, args.coord_coordinate_file, args.imod_circle_radius, args.imod_circle_thickness, args.imod_circle_color)
    else:
        for structure_name, _, _, _ in structures:
            shutil.move(f"{structure_name}.star", f"{structure_set_name}/")

    # Write structure details for each structure in the set
    with open(f"virtualice_{args.script_start_time}_info.txt", "a") as f:
        if f.tell() == 0:  # Only write the header line if the file is new
            f.write("structure_set_name structure_name mass(kDa) num_images num_particles_projected num_particles_saved\n")
        for structure_name, _, mass, _ in structures:
            # Write the correct number of saved particles for each structure
            f.write(f"{structure_set_name} {structure_name} {mass} {args.num_images} {structure_particles_projected[structure_name]} {structure_particles_saved[structure_name]}\n")

    # Return total projections, saved coordinates, list of box sizes, and individual structure counts
    return structure_particles_projected, structure_particles_saved, box_sizes

def clean_up(args, structure_set_name, structure_names):
    """
    Clean up files at the end of the script for multi-structure micrographs.

    :param Namespace args: The argument namespace containing all the user-specified command-line arguments.
    :param str structure_set_name: The name of the structure set directory.
    :param list structure_names: List of structure names in the current structure set.

    This function handles cleaning up .pdb, .mrc, and .star files for each structure in the set.
    If binning is enabled, it also moves or deletes non-binned files as necessary.
    """
    print_and_log("", logging.DEBUG)
    print_and_log(f"Cleaning up {structure_set_name}...", logging.DEBUG)

    for structure_name in structure_names:
        # Remove .pdb and .mrc files for each structure
        for ext in ['.pdb', '.mrc']:
            file_to_remove = f"{structure_name}{ext}"
            if os.path.exists(file_to_remove):
                print_and_log(f"Removing {file_to_remove}", logging.DEBUG)
                os.remove(file_to_remove)

        # Move .star files from the run directory to the structure set directory
        star_file = f"{structure_name}.star"
        if os.path.exists(star_file):
            print_and_log(f"Moving {star_file} to {structure_set_name}/", logging.DEBUG)
            shutil.move(star_file, structure_set_name)

        if args.binning > 1:
            bin_dir = f"{structure_set_name}/bin_{args.binning}/"
            star_file = f"{structure_name}_bin{args.binning}.star"
            if os.path.exists(star_file):
                print_and_log(f"Moving {star_file} to {bin_dir}/", logging.DEBUG)
                shutil.move(star_file, bin_dir)

    bin_dir = f"{structure_set_name}/bin_{args.binning}/"
    # Clean up any remaining .point files in the structure directories
    for directory in [structure_set_name, bin_dir]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.mod~') or file.endswith('.point'):
                    os.remove(os.path.join(directory, file))

    # Handle binning
    if not args.keep:
        if args.binning > 1:
            # Move binned files to the structure set directory and remove non-binned files
            for file in os.listdir(structure_set_name):
                if file.endswith(f"_bin{args.binning}.star"):
                    shutil.move(os.path.join(structure_set_name, file), bin_dir)
                if file.endswith(('.mrc', '.coord', '.mod')) and not file.endswith(f"_bin{args.binning}.mrc"):
                    os.remove(os.path.join(structure_set_name, file))

            # Move binned files from bin directory to structure set directory
            for file in os.listdir(bin_dir):
                if file.endswith(('.mrc', '.coord', '.mod', '.star')):
                    shutil.move(os.path.join(bin_dir, file), structure_set_name)

            # Remove the bin directory if it's empty
            if not os.listdir(bin_dir):
                print_and_log(f"Removing empty bin directory: {bin_dir}", logging.DEBUG)
                shutil.rmtree(bin_dir)

            star_file = f"{structure_set_name}/*_bin{args.binning}.star"
            for file_path in glob.glob(star_file):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print_an_log(f"Error removing {file_path}: {e}", logging.WARNING)

    print_and_log(f"Clean-up completed for {structure_set_name}.", logging.DEBUG)

def print_run_information(num_micrographs, structure_set_names, structure_sets, time_str, total_number_of_particles_projected,
                          total_number_of_particles_with_saved_coordinates, total_cropped_particles, crop_particles,
                          imod_coordinate_file, coord_coordinate_file, binning, keep, output_directory):
    """
    Print run information based on the provided inputs.

    :param int num_micrographs: The total number of micrographs generated.
    :param list structure_set_names: List of names of all of the structure sets.
    :param list structure_sets: List of structure sets, each containing structure names.
    :param str time_str: The string representation of the total generation time.
    :param int total_number_of_particles_projected: The total number of particles projected across all structure sets.
    :param int total_number_of_particles_with_saved_coordinates: Total number of particles saved to coordinate files across all structure sets.
    :param int total_cropped_particles: The total number of cropped particles across all structure sets.
    :param bool crop_particles: Whether or not particles were cropped.
    :param bool imod_coordinate_file: Whether IMOD .mod coordinate files were saved.
    :param bool coord_coordinate_file: Whether .coord files were saved.
    :param int binning: Factor by which micrographs were downsampled.
    :param bool keep: Whether or not unbinned data was kept.
    :param str output_directory: Output directory for all run files.
    """
    print_and_log("", logging.DEBUG)

    # Print the overall summary of the run
    total_structure_sets = len(structure_set_names)
    print_and_log(f"\n\033[1m{'-' * 100}\n{('VirtualIce Generation Summary').center(100)}\n{'-' * 100}\033[0m", logging.WARNING)
    print_and_log(f"Time to generate \033[1m{num_micrographs}\033[0m micrograph{'s' if num_micrographs != 1 else ''} "
                  f"from \033[1m{total_structure_sets}\033[0m structure set{'s' if total_structure_sets != 1 else ''}: "
                  f"\033[1m{time_str}\033[0m", logging.WARNING)

    # Print the total number of particles projected and saved to coordinate files, and cropped particles information if applicable
    print_and_log(f"Total: \033[1m{total_number_of_particles_projected}\033[0m particles projected, "
                  f"\033[1m{total_number_of_particles_with_saved_coordinates}\033[0m saved to coordinate files"
                  + (f", \033[1m{total_cropped_particles}\033[0m particles cropped" if crop_particles else ""), logging.WARNING)

    # Print the run directory
    print_and_log(f"Run directory: \033[1m{output_directory}/\033[0m", logging.WARNING)

    # Print information for each structure set
    for structure_set_name, structure_set in zip(structure_set_names, structure_sets):
        # Format the structure names in the set
        structure_names = ", ".join([s[0] if isinstance(s, list) else s for s in structure_set])
        print_and_log(f"Structure set: \033[1m{structure_set_name} ({structure_names})\033[0m", logging.WARNING)

    #Binning/Downsampling information
    if binning > 1:
        print_and_log(f"  - Binned by {binning} data." if not keep else f"  - Binned data in bin_{binning} sub-director{'y' if total_structure_sets == 1 else 'ies'}.", logging.WARNING)

    # STAR file information
    print_and_log("  - One .star file per structure is in the SS sub-directory.", logging.WARNING)

    # COORD file information
    if coord_coordinate_file:
        print_and_log("  - One (x, y) .coord file per micrograph is in the SS sub-directory.", logging.WARNING)

    # IMOD file information
    if imod_coordinate_file:
        print_and_log("  - One IMOD .mod file per micrograph is in the SS sub-directory.", logging.WARNING)
        print_and_log("    To view, run a command like this: \033[1m3dmod image.mrc image.mod\033[0m  (Replace with your files)", logging.WARNING)

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

def view_in_3dmod_async(micrograph_files, imod_coordinate_file, structure_names):
    """
    Open micrograph files in 3dmod asynchronously.

    :param list micrograph_files: List of micrograph file paths to open in 3dmod.
    :param bool imod_coordinate_file: Whether IMOD .mod coordinate files were saved.
    :param list structure_names: List of structure names associated with the micrograph.
    """
    print_and_log("", logging.DEBUG)

    # If there's only one micrograph and IMOD coordinate files are requested, open the .mod file for the first structure
    if len(micrograph_files) == 1 and imod_coordinate_file:
        # Extract the base micrograph file name without the extension
        base_micrograph = micrograph_files[0].rsplit('.', 1)[0]

        # Now use the structure_names list to truncate the additional structure names
        first_structure = structure_names[0][0] if isinstance(structure_names[0], list) else structure_names[0]

        # Find the position where the first structure name is appended in the basename
        # Recover the original basename (before additional structure names were appended)
        if f"_{first_structure}" in base_micrograph:
            original_basename = base_micrograph.split(f"_{first_structure}")[0]
            # Reconstruct the base micrograph with only the first structure name appended
            base_micrograph = f"{original_basename}_{first_structure}"

        # Now form the .mod file path for the first structure
        mod_file = [f"{base_micrograph}.mod"]

        # Launch 3dmod with the micrograph and the .mod file
        subprocess.run(["3dmod"] + micrograph_files + mod_file)
    else:
        # If there are multiple micrographs, open them without .mod files
        subprocess.run(["3dmod"] + micrograph_files)

    print_and_log("Opening micrographs in 3dmod...")

def main():
    """
    Main function: Loops over structure sets, generates micrographs, optionally crops particles,
    optionally opens micrographs in 3dmod asynchronously, and prints run time & output information.
    """
    start_time = time.time()
    start_time_formatted = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(start_time))

    # Parse user arguments, check them, and update them conditionally if necessary
    args = parse_arguments(start_time_formatted)

    # Initialize totals for eventual summary printing
    total_structure_sets = len(args.structures)
    total_number_of_particles_projected = 0
    total_number_of_particles_with_saved_coordinates = 0
    total_cropped_particles = 0

    # Store structure set names, tasks (ie. results), and the corresponding box sizes
    structure_set_names = []
    structure_set_box_sizes = []  # Store box sizes for each structure in each set
    tasks = []

    # Process each structure set in parallel (each set contains one or more structures)
    with ProcessPoolExecutor(max_workers=args.parallelize_structures) as executor:
        for structure_set_index, structure_set in enumerate(args.structures):
            # Submit task for each structure set
            task = executor.submit(generate_micrographs, args, structure_set, structure_set_index, total_structure_sets)
            tasks.append((task, structure_set))  # Store task and associated structure set

        # Wait for all tasks to complete and aggregate results
        for task, structure_set in tasks:
            structure_set_name = f"structure_set_{tasks.index((task, structure_set)) + 1}"  # Name as structure_set_1, structure_set_2, etc.
            structure_set_names.append(structure_set_name)
            structure_particles_projected, structure_particles_saved, box_sizes = task.result()

            # Store box sizes for further particle cropping
            structure_set_box_sizes.append(box_sizes)

            # Aggregate totals
            total_number_of_particles_projected += sum(structure_particles_projected.values())
            total_number_of_particles_with_saved_coordinates += sum(structure_particles_saved.values())

    # Open 3dmod asynchronously in a separate thread so cropping can happen simultaneously
    if args.view_in_3dmod:
        micrograph_files = find_micrograph_files(structure_set_names)
        if micrograph_files:
            threading.Thread(target=view_in_3dmod_async, args=(micrograph_files, args.imod_coordinate_file, args.structures[0])).start()

    # Crop particles if requested
    if args.crop_particles:
        with ProcessPoolExecutor(max_workers=args.parallelize_structures) as executor:
            crop_tasks = []
            for structure_set_name, box_sizes in zip(structure_set_names, structure_set_box_sizes):
                for i, structure_name in enumerate(args.structures[structure_set_names.index(structure_set_name)]):
                    # Use the appropriate box_size for each structure
                    box_size = args.box_size if args.box_size is not None else box_sizes[i]
                    structure_name_str = structure_name[0] if isinstance(structure_name, list) else structure_name  # Converts element of structure set to structure name, if necessary
                    crop_task = executor.submit(crop_particles_from_micrographs, structure_name_str, structure_set_name, box_size, args.apix, args.max_crop_particles, args.cpus, args.use_gpu, args.gpu_ids)
                    crop_tasks.append(crop_task)

            for crop_task in crop_tasks:
                total_cropped_particles += crop_task.result()

    # Clean up and finalize each structure set
    for structure_set_name, structure_set in zip(structure_set_names, args.structures):
        # Extract structure names from the structure set (flattened, if needed)
        structure_names = [structure[0] if isinstance(structure, list) else structure for structure in structure_set]
        clean_up(args, structure_set_name, structure_names)

    # Calculate total number of micrographs generated
    num_micrographs = args.num_images * len(structure_set_names)

    # Calculate run time
    end_time = time.time()
    time_str = time_diff(end_time - start_time)

    # Print summary of the run
    print_run_information(num_micrographs, structure_set_names, args.structures, time_str, total_number_of_particles_projected,
                          total_number_of_particles_with_saved_coordinates, total_cropped_particles, args.crop_particles,
                          args.imod_coordinate_file, args.coord_coordinate_file, args.binning, args.keep, args.output_directory)

if __name__ == "__main__":
    main()
