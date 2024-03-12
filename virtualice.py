#!/usr/bin/env python3
#
# Author: Alex J. Noble with help from GPT4, 2023-24 @SEMC, under the MIT License
#
# VirtualIce: Synthetic CryoEM Micrograph Generator
#
# This script generates synthetic cryoEM micrographs given protein structures and a list of
# noise micrographs and their corresponding defoci. It is intended that the noise micrographs
# are cryoEM images of buffer and that the junk & substrate are masked out using AnyLabeling.
#
# Dependencies: EMAN2 installation (specifically e2pdb2mrc.py, e2project3d.py, e2proc3d.py, and e2proc2d.py)
#               pip install mrcfile numpy scipy matplotlib cv2 SimpleITK
#
# This program depends on EMAN2 to function properly. Users must separately
# install EMAN2 to use this program.

# If the user wishes to output IMOD coordinate files, then IMOD needs to be
# installed separately.
#
# EMAN2 is distributed under a dual license - BSD-3-Clause and GPL-2.0.
# For the details of EMAN2's licensing, please refer to:
# - BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
# - GPL-2.0: https://opensource.org/licenses/GPL-2.0
#
# You can obtain the EMAN2 source code from its official GitHub repository:
# https://github.com/cryoem/eman2
#
# IMOD is distributed under GPL-2.0. For details, see the link above.
#
# You can obtain the IMOD source code and packages from its official website:
# https://bio3d.colorado.edu/imod/
#
# Ensure compliance with EMAN2's and IMOD's license terms when obtaining and using them.
__version__ = "1.0.0"

import os
import re
import cv2
import glob
import gzip
import json
import time
import random
import shutil
import inspect
import logging
import mrcfile
import argparse
import textwrap
import itertools
import subprocess
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.path import Path
from multiprocessing import Pool
from urllib import request, error
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Global variable to store verbosity level
global_verbosity = 0

def parse_arguments():
    """
    Parses command-line arguments.

    :returns argparse.Namespace: An object containing attributes for each command-line argument.
    """
    parser = argparse.ArgumentParser(description="VirtualIce: A feature-rich synthetic cryoEM micrograph generator that projects pdbs|mrcs onto existing buffer cryoEM micrographs. Star files for particle coordinates are outputed by default, mod and coord files are optional. Particle coordinates located within per-micrograph polygons at junk/substrate locations are projected but not written to coordinate files.",
    epilog="""
    Examples:
      1. Basic usage: virtualice.py -s 1TIM -n 10
         Generates 10 random micrographs of PDB 1TIM.

      2. Advanced usage: virtualice.py -s 1TIM r my_structure.mrc 11638 rp -n 3 -I -P -J -Q 90 -b 4 -D n -j 2 -C
         Generates 3 random micrographs of PDB 1TIM, a random EMDB/PDB structure, a local structure called my_structure.mrc, EMD-11638, and a random PDB.
         Outputs an IMOD .mod coordinate file, png, and jpeg (quality 90) for each micrograph, and bins all images by 4.
         Uses a non-random distribution of particles, parallelizes micrograph generation across 2 CPUs, and crops particles
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter)  # Preserves whitespace for better formatting

    # Input Options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument("-s", "--structures", type=str, nargs='+', default=['1TIM', '11638', 'r'], help="PDB ID(s), EMDB ID(s), names of local .pdb or .mrc/.map files, 'r' or 'random' for a random PDB or EMDB map, 'rp' for a random PDB, and/or 're' or 'rm' for a random EMDB map. Local .mrc/.map files must have voxel size in the header so that they are scaled properly. Separate structures with spaces. Default is %(default)s.")
    input_group.add_argument("-i", "--image_list_file", type=str, default="ice_images/good_images_with_defocus.txt", help="File containing local filenames of images with a defocus value after each filename (space between). Default is '%(default)s'.")
    input_group.add_argument("-d", "--image_directory", type=str, default="ice_images", help="Local directory name where the micrographs are stored in mrc format. They need to be accompanied with a text file containing image names and defoci (see --image_list_file). Default directory is %(default)s")

    # Particle and Micrograph Generation Options
    particle_micrograph_group = parser.add_argument_group('Particle and Micrograph Generation Options')
    particle_micrograph_group.add_argument("-n", "--num_images", type=int, default=5, help="Number of micrographs to create for each structure requested. Default is %(default)s")
    particle_micrograph_group.add_argument("-N", "--num_particles", type=check_num_particles, help="Number of particles to project onto the micrograph after rotation. Input an integer or 'max'. Default is a random number (weighted to favor numbers above 100 twice as much as below 100) up to a maximum of the number of particles that can fit into the micrograph without overlapping.")
    particle_micrograph_group.add_argument("-a", "--apix", type=float, default=1.096, help="Pixel size of the ice images, used to scale pdbs during pdb>mrc conversion (EMAN2 e2pdb2mrc.py option). Default is %(default)s (the pixel size of the ice images used during development)")
    particle_micrograph_group.add_argument("-r", "--pdb_to_mrc_resolution", type=float, default=3, help="Resolution in Angstroms for PDB to MRC conversion (EMAN2 e2pdb2mrc.py option). Default is %(default)s")
    particle_micrograph_group.add_argument("-T", "--std_threshold", type=float, default=-1.0, help="Threshold for removing noise in terms of standard deviations above the mean. Default is %(default)s")
    particle_micrograph_group.add_argument("-f", "--num_simulated_particle_frames", type=int, default=50, help="Number of simulated particle frames to generate Poisson noise. Default is %(default)s")
    particle_micrograph_group.add_argument("-G", "--scale_percent", type=float, default=33.33, help="How much larger to make the resulting mrc file from the pdb file compared to the minimum equilateral cube. Extra space allows for more delocalized CTF information (default: %(default)s; ie. %(default)s%% larger)")
    particle_micrograph_group.add_argument("-D", "--distribution", type=str, choices=['r', 'random', 'n', 'non-random'], default=None, help="Distribution type for generating particle locations: 'random' (or 'r') and 'non-random' (or 'n'). Random is a random selection from a uniform 2D distribution. Non-random selects from 4 distributions: 1) Mimicking the micrograph ice thickness (darker areas = more particles), 2) Gaussian clumps, 3) circular, and 4) inverse circular. Default is %(default)s which selects a distribution per micrograph based on internal weights.")
    particle_micrograph_group.add_argument("-B", "--border", type=int, default=-1, help="Minimum distance of center of particles from the image border. Default is  %(default)s = reverts to half boxsize")
    particle_micrograph_group.add_argument("--edge_particles", action="store_true", help="Allow particles to be placed up to the edge of the micrograph.")
    particle_micrograph_group.add_argument("--save_edge_coordinates", action="store_true", help="Save particle coordinates that are closer than half a particle box size from the edge, requires --edge_particles to be True or --border to be less than half the particle box size.")

    # Simulation Options
    simulation_group = parser.add_argument_group('Simulation Options')
    simulation_group.add_argument("-m", "--min_ice_thickness", type=float, default=30, help="Minimum ice thickness, which scales how much the particle is added to the image (this is a relative value)")
    simulation_group.add_argument("-M", "--max_ice_thickness", type=float, default=90, help="Maximum ice thickness, which scales how much the particle is added to the image (this is a relative value)")
    simulation_group.add_argument("-t", "--ice_thickness", type=float, help="Request a specific ice thickness, which scales how much the particle is added to the image (this is a relative value). This will override --min_ice_thickness and --max_ice_thickness")
    simulation_group.add_argument("-p", "--preferred_orientation", action="store_true", help="Enable preferred orientation mode")
    simulation_group.add_argument("-E", "--fixed_euler_angle", type=float, default=0.0, help="Fixed Euler angle for preferred orientation mode (usually 0 or 90 degrees) (EMAN2 e2project3d.py option)")
    simulation_group.add_argument("--orientgen_method", type=str, default="even", choices=["eman", "even", "opt", "saff"], help="Orientation generator method to use for preferred orientation (EMAN2 e2project3d.py option). Default is %(default)s")
    simulation_group.add_argument("-A", "--delta_angle", type=float, default=13.1, help="The angular separation of preferred orientations in degrees for non-fixed angles. Default is a number that doesn't cause aliasing after 360 degrees")
    simulation_group.add_argument("-F", "--phitoo", type=float, default=0.1, help="Phitoo value for random 3D projection (ie. no preferred orientation) (EMAN2 e2project3d.py option). This is the angular step size for rotating before projecting. Default is %(default)s")
    simulation_group.add_argument("--ampcont", type=float, default=10, help="Amplitude contrast percentage when applying CTF to projections (EMAN2 e2proc2d.py option). Default is %(default)s (ie. 10%%)")
    simulation_group.add_argument("--Cs", type=float, default=0.001, help="Microscope spherical aberration when applying CTF to projections (EMAN2 e2proc2d.py option). Default is %(default)s because the microscope used to collect the provided buffer cryoEM micrographs has a Cs corrector")
    simulation_group.add_argument("-K", "--voltage", type=float, default=300, help="Microscope voltage (keV) when applying CTF to projections (EMAN2 e2proc2d.py option). Default is %(default)s")

    # Junk Labels Options
    junk_labels_group = parser.add_argument_group('Junk Labels Options')
    junk_labels_group.add_argument("--no_junk_filter", action="store_true", help="Turn off junk filtering; i.e. Do not remove particles from coordinate files that are on/near junk or substrate.")
    junk_labels_group.add_argument("-S", "--json_scale", type=int, default=4, help="Binning factor used when labeling junk to create the JSON files with AnyLabeling. Default is %(default)s")
    junk_labels_group.add_argument("-x", "--flip_x", action="store_true", help="Flip the polygons that identify junk along the x-axis")
    junk_labels_group.add_argument("-y", "--flip_y", action="store_true", help="Flip the polygons that identify junk along the y-axis")
    junk_labels_group.add_argument("-e", "--polygon_expansion_distance", type=int, default=5, help="Number of pixels to expand each polygon in the JSON file that defines areas to not place particle coordinates. The size of the pixels used here is the same size as the pixels that the JSON file uses (ie. the binning used when labeling the micrographs in AnyLabeling). Default is %(default)s")

    # Particle Cropping Options
    particle_cropping_group = parser.add_argument_group('Particle Cropping Options')
    particle_cropping_group.add_argument("-C", "--crop_particles", action="store_true", help="Enable cropping of particles from micrographs. Particles will be extracted to the [structure_name]/Particles/ directory as .mrc files. Default is no cropping.")
    particle_cropping_group.add_argument("-X", "--box_size", type=int, default=None, help="Box size for cropped particles (x and y dimensions are the same). Particles with box sizes that fall outside the micrograph will not be cropped. Default is the size of the mrc used for particle projection after internal preprocessing.")

    # Micrograph Output Options
    output_group = parser.add_argument_group('Micrograph Output Options')
    output_group.add_argument("-o", "--output_directory", type=str, help="Directory to save all output files. If not specified, a unique directory will be created.")
    output_group.add_argument("--mrc", action="store_true", default=True, help="Save micrographs as .mrc (default if no format is specified)")
    output_group.add_argument("--no_mrc", dest="mrc", action="store_false", help="Do not save micrographs as .mrc")
    output_group.add_argument("-P", "--png", action="store_true", help="Save micrographs as .png")
    output_group.add_argument("-J", "--jpeg", action="store_true", help="Save micrographs as .jpeg")
    output_group.add_argument("-Q", "--jpeg-quality", type=int, default=95, help="Quality of saved .jpeg images (0 to 100). Default is %(default)s")
    output_group.add_argument("-b", "--binning", type=check_binning, default=1, help="Bin/Downsample the micrographs by Fourier cropping after superimposing particle projections. Binning is the sidelength divided by this factor (e.g. -b 4 for a 4k x 4k micrograph will result in a 1k x 1k micrograph) (e.g. -b 1 is unbinned). Default is %(default)s")
    output_group.add_argument("-k", "--keep", action="store_true", help="Keep the non-downsampled micrographs if downsampling is requested. Non-downsampled micrographs are deleted by default")
    output_group.add_argument("-I", "--imod_coordinate_file", action="store_true", help="Also output one IMOD .mod coordinate file per micrograph. Note: IMOD must be installed and working")
    output_group.add_argument("-O", "--coord_coordinate_file", action="store_true", help="Also output one .coord coordinate file per micrograph")

    # System and Program Options
    misc_group = parser.add_argument_group('System and Program Options')
    misc_group.add_argument("-c", "--cpus", type=int, default=os.cpu_count(), help="Number of CPUs to use for various processing steps. Default is the number of CPU cores available: %(default)s")
    misc_group.add_argument("-j", "--parallel_processes", type=int, default=1, help="Maximum number of parallel processes for micrograph generation. Each parallel process will use up to '--cpus' number of CPU cores for various steps. Default is %(default)s")
    misc_group.add_argument("-V", "--verbosity", type=int, default=1, help="Set verbosity level: 0 (quiet), 1 (some output), 2 (verbose), 3 (debug). For 0-2, a log file will be additionally written with 2. For 3, a log file will be additionally written with 3. Default is %(default)s")
    misc_group.add_argument("-q", "--quiet", action="store_true", help="Set verbosity to 0 (quiet). Overrides --verbosity if both are provided")
    misc_group.add_argument("-v", "--version", action="version", help="Show version number and exit", version=f"VirtualIce v{__version__}")

    args = parser.parse_args()

    # Set verbosity level
    args.verbosity = 0 if args.quiet else args.verbosity

    # Setup logging based on the verbosity level
    setup_logging(args.verbosity)

    if args.crop_particles and not args.mrc:
        args.mrc = True
        print_and_log(f"Notice: Since cropping (--crop_particles) is requested, then --mrc must be turned on. --mrc is now set to True.", logging.INFO)

    if not (args.mrc or args.png or args.jpeg):
        parser.error("No format specified for saving images. Please specify at least one format.")

    # Determine output directory
    if not args.output_directory:
        # Create a unique directory name using the current date and time
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        args.output_directory = f"VirtualIce_run_{current_time}"

    # Create the output directory if it doesn't exist
    #if not os.path.exists(args.output_directory):
    #    os.makedirs(args.output_directory)

    # Print all arguments for the user's information
    formatted_output = ""
    for arg, value in vars(args).items():
        formatted_output += f"{arg}: {value};\n"
    argument_printout = textwrap.fill(formatted_output, width=80)  # Wrap the output text to fit in rows and columns

    print_and_log("-----------------------------------------------------------------------------------------------", logging.WARNING)
    print_and_log(f"Generating {args.num_images} synthetic micrographs for each structure ({args.structures}) using micrographs in {args.image_directory.rstrip('/')}/ ...\n", logging.WARNING)
    print_and_log("VirtualIce arguments:\n", logging.WARNING)
    print_and_log(argument_printout, logging.WARNING)
    print_and_log("-----------------------------------------------------------------------------------------------\n", logging.WARNING)

    return args

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
    :return int: Value if it is valid.
    :raises ArgumentTypeError: If the value is not in the allowed range.
    """
    ivalue = int(value)
    if ivalue < 2 or ivalue >= 64:
        raise argparse.ArgumentTypeError("Binning must be between 2 and 64")
    return ivalue

def setup_logging(verbosity):
    """
    Sets up logging configuration for both console and file output based on the specified verbosity level.

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
    logging_level = levels.get(verbosity, logging.INFO)

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_filename = f"virtualice_{datetime_str}.log"

    simple_formatter = logging.Formatter('%(message)s')
    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d, %(funcName)s)')

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(detailed_formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    if verbosity < 3:
        ch.setFormatter(simple_formatter)
    else:
        ch.setFormatter(detailed_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

def print_and_log(message, level=logging.INFO):
    """
    Prints and logs a message with the specified level, including detailed debug information if verbosity is set to 3.

    :param str message: The message to print and log.
    :param int level: The logging level for the message (e.g., logging.INFO, logging.DEBUG).

    If verbosity is set to 3, the function logs additional details about the caller,
    including module name, function name, line number, and function parameters.
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

def time_diff(time_diff):
    """
    Convert the time difference to a human-readable format.

    :param float time_diff: The time difference in seconds.
    :return str: A formatted string indicating the time difference.
    """
    print_and_log("", logging.DEBUG)
    # Convert the time difference to a timedelta object
    delta = timedelta(seconds=time_diff)
    # Format the timedelta object based on its components
    if delta.days > 0:
        # If the time difference is more than a day, display days, hours, minutes, and seconds
        time_str = str(delta)
    elif delta.seconds >= 3600:
        # If the time difference is less than a day, display hours, minutes, and seconds
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours} hours, {minutes} minutes, {seconds} seconds"
    else:
        # If the time difference is less than an hour, display minutes and seconds
        minutes, seconds = divmod(delta.seconds, 60)
        time_str = f"{minutes} minutes, {seconds} seconds"

    return time_str

def is_local_pdb_path(input_str):
    """
    Check if the input string is a path to a local PDB file.

    :param str input_str: The input string to be checked.
    :return bool: True if the input string is a valid path to a local `.pdb` file, False otherwise.
    """
    print_and_log("", logging.DEBUG)
    return os.path.isfile(input_str) and input_str.endswith('.pdb')

def is_local_mrc_path(input_str):
    """
    Check if the input string is a path to a local MRC/Map file.

    :param str input_str: The input string to be checked.
    :return bool: True if the input string is a valid path to a local `.mrc` or `.map` file, False otherwise.
    """
    print_and_log("", logging.DEBUG)
    return os.path.isfile(input_str) and (input_str.endswith('.mrc') or input_str.endswith('.map'))

def is_emdb_id(input_str):
    """
    Check if the input string structured as a valid EMDB ID.

    :param str input_str: The input string to be checked.
    :return bool: True if the input string is a valid EMDB ID format, False otherwise.
    """
    print_and_log("", logging.DEBUG)
    return input_str.isdigit() and (len(input_str) == 4 or len(input_str) == 5)

def is_pdb_id(structure_input):
    """
    Check if the input string is a valid PDB ID.

    :param str structure_input: The input string to be checked.
    :return bool: True if the input string is a valid PDB ID, False otherwise.
    """
    print_and_log("", logging.DEBUG)
    # PDB ID must be 4 characters: first character is a number, next 3 are alphanumeric, and there must be at least one letter
    return bool(re.match(r'^[0-9][A-Za-z0-9]{3}$', structure_input) and any(char.isalpha() for char in structure_input))

def process_structure_input(structure_input, std_devs_above_mean, pixelsize):
    """
    Process each structure input by identifying whether it's a PDB ID for download, EMDB ID for download, a local file path, or a request for a random structure.
    Normalize any input .map/.mrc file and convert to .mrc.

    :param str structure_input: The structure input which could be a PDB ID, EMDB ID, a local file path, a request for a random PDB/EMDB structure ('r' or 'random'), a request for a random PDB structure ('rp'), a request for a random EMDB structure ('re' or 'rm').
    :param float std_devs_above_mean: Number of standard deviations above the mean to threshold downloaded/imported .mrc/.map files (for getting rid of some dust).
    :param float pixelsize: Pixel size of the micrograph onto which mrcs will be projected. Used to scale downloaded/imported .pdb/.mrc/.map files.
    :return tuple: A tuple containing the structure ID and file type if the file is successfully identified, downloaded, or a random structure is selected; None if there was an error or the download failed.
    """
    print_and_log("", logging.DEBUG)
    def process_local_mrc_file(file_path):
        converted_file = normalize_and_convert_mrc(file_path)
        threshold_mrc_file(f"{converted_file}.mrc", std_devs_above_mean)
        scale_mrc_file(f"{converted_file}.mrc", pixelsize)
        converted_file = normalize_and_convert_mrc(f"{converted_file}.mrc")
        return (converted_file, "mrc") if converted_file else None

    def download_random_pdb_structure():
        print_and_log("Downloading a random PDB...", logging.INFO)
        pdb_id = download_random_pdb()
        return (pdb_id, "pdb") if pdb_id else None

    def download_random_emdb_structure():
        print_and_log("Downloading a random EMDB map...", logging.INFO)
        emdb_id = download_random_emdb()
        structure_input = f"emd_{emdb_id}.map"
        return process_local_mrc_file(structure_input) if emdb_id else None

    if structure_input.lower() in ['r', 'random']:
        if random.choice(["pdb", "emdb"]) == "pdb":
            return download_random_pdb_structure()
        else:
            return download_random_emdb_structure()
    elif structure_input.lower() == 'rp':
        return download_random_pdb_structure()
    elif structure_input.lower() == 're' or structure_input.lower() == 'rm':
        return download_random_emdb_structure()
    elif is_local_pdb_path(structure_input):
        print_and_log(f"Using local PDB file: {structure_input}", logging.WARNING)
        # Make a local copy of the file
        if not os.path.samefile(structure_input, os.path.basename(structure_input)):
            shutil.copy(structure_input, os.path.basename(structure_input))
        return (os.path.basename(structure_input).split('.')[0], "pdb")
    elif is_local_mrc_path(structure_input):
        print_and_log(f"Using local MRC/MAP file: {structure_input}", logging.WARNING)
        # Make a local copy of the file
        if not os.path.samefile(structure_input, os.path.basename(structure_input)):
            shutil.copy(structure_input, os.path.basename(structure_input))
        return process_local_mrc_file(structure_input)
    elif is_emdb_id(structure_input):
        if download_emdb(structure_input):
            structure_input = f"emd_{structure_input}.map"
            return process_local_mrc_file(structure_input)
        else:
            return None
    elif is_pdb_id(structure_input):
        if download_pdb(structure_input):
            return (structure_input, "pdb")
        else:
            print_and_log(f"Failed to download PDB: {structure_input}. Please check the ID and try again.", logging.WARNING)
            return None
    else:
        print_and_log(f"Unrecognized structure input: {structure_input}. Please enter a valid PDB ID, EMDB ID, local file path, or 'random'.", logging.WARNING)
        return None

def download_pdb(pdb_id, suppress_errors=False):
    """
    Download a PDB file from the RCSB website.

    :param str pdb_id: The ID of the PDB to be downloaded.
    :param bool suppress_errors: If True, suppress error messages. Useful for random PDB downloads.
    :return bool: True if the PDB exists, False if it doesn't.
    """
    print_and_log("", logging.DEBUG)
    if not suppress_errors:
        print_and_log(f"Downloading PDB {pdb_id}...", logging.INFO)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        request.urlretrieve(url, f"{pdb_id}.pdb")
        print_and_log(f"Done!\n", logging.INFO)
        return True
    except error.HTTPError as e:
        if not suppress_errors:
            print_and_log(f"Failed to download PDB {pdb_id}. HTTP Error: {e.code}\n", logging.WARNING)
        return False
    except Exception as e:
        if not suppress_errors:
            print_and_log(f"An unexpected error occurred while downloading PDB {pdb_id}. Error: {e}\n", logging.WARNING)
        return False

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
        # No need to explicitly handle failure; loop continues until a successful download occurs

def download_emdb(emdb_id, suppress_errors=False):
    """
    Download and decompress an EMDB map file using urllib.

    :param str emdb_id: The ID of the EMDB map to be downloaded.
    :param bool suppress_errors: If True, suppress error messages. Useful for random PDB downloads.
    :return bool: True if the map exists and is downloaded, False if not.
    """
    print_and_log("", logging.DEBUG)
    url = f"https://files.wwpdb.org/pub/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz"
    local_filename = f"emd_{emdb_id}.map.gz"

    try:
        # Download the gzipped map file
        if not suppress_errors:
            print_and_log(f"Downloading EMD-{emdb_id}...", logging.INFO)
        request.urlretrieve(url, local_filename)

        # Decompress the downloaded file
        with gzip.open(local_filename, 'rb') as f_in:
            with open(local_filename.replace('.gz', ''), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove the compressed file after decompression
        os.remove(local_filename)

        if not suppress_errors:
            print_and_log(f"Download and decompression complete for EMD-{emdb_id}.", logging.INFO)
        return True
    except error.HTTPError as e:
        if not suppress_errors:
            print_and_log(f"EMD-{emdb_id} not found. HTTP Error: {e.code}", logging.WARNING)
        return False
    except Exception as e:
        if not suppress_errors:
            print_and_log(f"An unexpected error occurred while downloading EMD-{emdb_id}. Error: {e}", logging.WARNING)
        return False

def download_random_emdb():
    """
    Download a random EMDB map by trying random IDs with urllib.

    :return str: The ID of the EMDB map if downloaded successfully, otherwise False.
    """
    print_and_log("", logging.DEBUG)
    while True:
        # Generate a random EMDB ID within a reasonable range
        emdb_id = str(random.randint(1, 43542)).zfill(4)  # Makes a 4 or 5 digit number with leading zeros. Random 1-3 digits will also be 4 digit.
        success = download_emdb(emdb_id, suppress_errors=True)
        if success:
            return emdb_id

def normalize_and_convert_mrc(input_file):
    """
    Normalize and, if necessary, pad the .map/.mrc file to make all dimensions equal, centering the original volume.

    This function first normalizes the input MRC or MAP file using the `e2proc3d.py` script from EMAN2,
    ensuring the mean edge value is normalized. If the volume dimensions are not equal, it calculates
    the necessary padding to make the dimensions equal, with the original volume centered within the new
    dimensions. The adjusted volume is saved to the output file specified by the input file name or
    altered to have a '.mrc' extension if necessary.

    :param str input_file: Path to the input MRC or MAP file.
    :returns str: The base name of the output MRC file, without the '.mrc' extension, or None if an error occurred.

    Note:
    - The function attempts to remove the original input file if it's different from the output file to avoid duplication.
    - A temporary file ('temp_normalized.mrc') is used during processing for normalization.
    - If the input file is not a cube (i.e., all dimensions are not equal), the function calculates the padding needed to center the volume within a cubic volume whose dimension is equal to the maximum dimension of the original volume.
    - The volume is padded with the average value found in the original data, ensuring that added regions do not introduce artificial density.
    """
    print_and_log("", logging.DEBUG)
    with mrcfile.open(input_file, mode='r') as mrc:
        dims = mrc.data.shape
        max_dim = max(dims)

    # Calculate the padding needed to make all dimensions equal to the max dimension
    padding = [(max_dim - dim) // 2 for dim in dims]
    pad_width = [(pad, max_dim - dim - pad) for pad, dim in zip(padding, dims)]

    output_file = input_file if input_file.endswith('.mrc') else input_file.rsplit('.', 1)[0] + '.mrc'

    # Temporary file for normalized volume
    normalized_file = "temp_normalized.mrc"

    try:
        # Normalize the volume
        output = subprocess.run(["e2proc3d.py", input_file, normalized_file, "--outtype=mrc", "--process=normalize.edgemean"], capture_output=True, text=True, check=True)
        print_and_log(output, logging.INFO)

        # Read the normalized volume, pad it, and save to the output file
        with mrcfile.open(normalized_file, mode='r+') as mrc:
            data_padded = np.pad(mrc.data, pad_width, mode='constant', constant_values=np.mean(mrc.data))
            mrc.set_data(data_padded)
            mrc.update_header_from_data()
            mrc.close()

        # Move the padded, normalized file to the desired output file location
        if normalized_file != output_file:
            os.rename(normalized_file, output_file)

        # Remove the original .map file if it's different from the output file
        if input_file != output_file and os.path.exists(input_file):
            os.remove(input_file)
    except subprocess.CalledProcessError as e:
        if os.path.exists(normalized_file):
            os.remove(normalized_file)
        return None

    return output_file.rsplit('.', 1)[0]

def threshold_mrc_file(input_file_path, std_devs_above_mean):
    """
    Thresholds an MRC file so that all voxel values below a specified number of 
    standard deviations above the mean are set to zero.

    :param str input_file_path: Path to the input MRC file.
    :param float std_devs_above_mean: Number of standard deviations above the mean for thresholding.
    :param str output_file_path: Path to the output MRC file. If None, overwrite the input file.
    """
    print_and_log("", logging.DEBUG)
    with mrcfile.open(input_file_path, mode='r+') as mrc:
        data = mrc.data
        mean = data.mean()
        std_dev = data.std()
        threshold = mean + (std_devs_above_mean * std_dev)
        data[data < threshold] = 0  # Set values below threshold to zero
        mrc.set_data(data)  # Update the MRC file with thresholded data

def scale_mrc_file(input_file, pixelsize):
    """
    Scale an MRC file to a specified pixel size, allowing both upscaling and downscaling.

    :param str input_mrc_path: Path to the input MRC file.
    :param float pixelsize: The desired pixel size in Angstroms.
    """
    print_and_log("", logging.DEBUG)
    # Read the current voxel size
    with mrcfile.open(input_file, mode='r') as mrc:
        original_voxel_size = mrc.voxel_size.x  # Assuming cubic voxels for simplicity
        original_shape = mrc.data.shape

    # Calculate the scale factor
    scale_factor = original_voxel_size / pixelsize

    # Calculate the new dimensions and round down to the next integer that is evenly divisible by 2 for future FFT processing
    scaled_dimension_x = int(((original_shape[0] * scale_factor) // 2) * 2) 
    scaled_dimension_y = int(((original_shape[1] * scale_factor) // 2) * 2)
    scaled_dimension_z = int(((original_shape[2] * scale_factor) // 2) * 2)

    # Construct the e2proc3d.py command for scaling. Using a temp file because otherwise the mrc filesize and header don't match, causing a warning from mrcfile during thresholding
    if scale_factor < 1:
        command = ["e2proc3d.py",
            input_file, f"temp_scale_{input_file}",
            "--scale={}".format(scale_factor),
            "--clip={},{},{}".format(scaled_dimension_x, scaled_dimension_y, scaled_dimension_z)]
    elif scale_factor > 1:
        command = ["e2proc3d.py",
            input_file, f"temp_scale_{input_file}",
            "--clip={},{},{}".format(scaled_dimension_x, scaled_dimension_y, scaled_dimension_z),
            "--scale={}".format(scale_factor)]
    else:  # scale_factor == 1:
        return

    try:
        output = subprocess.run(command, capture_output=True, text=True, check=True)
        os.system(f"mv temp_scale_{input_file} {input_file}")
        print_and_log(output, logging.INFO)
    except subprocess.CalledProcessError as e:
        print_and_log(f"Error during scaling operation: {e}", logging.WARNING)

def convert_pdb_to_mrc(pdb_name, apix, res):
    """
    Convert a PDB file to MRC format using EMAN2's e2pdb2mrc.py script.

    :param str pdb_name: The name of the PDB to be converted.
    :param float apix: The pixel size used in the conversion.
    :param int res: The resolution to be used in the conversion.

    :return int: The mass extracted from the e2pdb2mrc.py script output.
    """
    print_and_log("", logging.DEBUG)
    print_and_log(f"Converting PDB {pdb_name} to MRC using EMAN2's e2pdb2mrc.py...", logging.INFO)
    cmd = ["e2pdb2mrc.py", "--apix", str(apix), "--res", str(res), "--center", f"{pdb_name}.pdb", f"{pdb_name}.mrc"]
    output = subprocess.run(cmd, capture_output=True, text=True)
    print_and_log(output, logging.INFO)
    try:
        # Attempt to extract the mass from the output
        mass = int([line for line in output.stdout.split("\n") if "mass of" in line][0].split()[-2])
    except IndexError:
        # If the mass is not found in the output, set it to 0 and print a warning
        mass = 0
        print_and_log(f"Warning: Mass not found for PDB {pdb_name}. Setting mass to 0.", logging.WARNING)
    print_and_log(f"Done!\n", logging.INFO)
    return mass

def readmrc(mrc_path):
    """
    Read an MRC file and return its data as a NumPy array.

    :param str mrc_path: The file path of the MRC file to read.
    :return numpy_array: The data of the MRC file as a NumPy array.
    """
    print_and_log("", logging.DEBUG)
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data
        numpy_array = np.array(data)

    return numpy_array

def writemrc(mrc_path, numpy_array):
    """
    Write a NumPy array as an MRC file.

    :param strmrc_path: The file path of the MRC file to write.
    :param numpy_array: The NumPy array to be written.
    """
    print_and_log("", logging.DEBUG)
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(numpy_array)

    return

def write_star_header(file_basename, apix, voltage, cs):
    """
    Write the header for a .star file.

    :param str file_basename: The basename of the file to which the header should be written.
    :param float apix: The pixel size used in the conversion.
    :param float voltage: The voltage used in the conversion.
    :param float cs: The spherical aberration used in the conversion.
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
        star_file.write('_rlnOpticsGroup #5\n')

def write_all_coordinates_to_star(structure_name, image_path, particle_locations):
    """
    Write all particle locations to a STAR file.

    :param str structure_name: The name of the structure file.
    :param str image_path: The path of the image to add.
    :param list_of_tuples particle_locations: A list of tuples, where each tuple contains the x, y coordinates.
    """
    print_and_log("", logging.DEBUG)
    # Open the star file once and write all coordinates
    with open(f'{structure_name}.star', 'a') as star_file:
        for location in particle_locations:
            x_shift, y_shift = location
            star_file.write(f'{image_path} {x_shift} {y_shift} 0 1\n')

def convert_point_to_model(point_file, output_file):
    """
    Write an IMOD .mod file with particle coordinates.

    :param str point_file: Path to the input .point file.
    :param str output_file: Output file path for the .mod file.
    """
    print_and_log("", logging.DEBUG)
    try:
        # Run point2model command and give particles locations a circle of radius 3. Adjust the path if point2model is located elsewhere on your system.
        output = subprocess.run(["point2model", "-circle", "3", "-scat", point_file, output_file], capture_output=True, text=True, check=True)
        print_and_log(output, logging.INFO)
    except subprocess.CalledProcessError:
        print_and_log("Error while converting coordinates using point2model.", logging.WARNING)
    except FileNotFoundError:
        print_and_log("point2model not found. Ensure IMOD is installed and point2model is in your system's PATH.", logging.WARNING)

def write_mod_file(coordinates, output_file):
    """
    Write an IMOD .mod file with particle coordinates.

    :param list_of_tuples coordinates: List of (x, y) coordinates for the particles.
    :param str output_file: Output file path for the .mod file.
    """
    print_and_log("", logging.DEBUG)
    # Step 1: Write the .point file
    point_file = os.path.splitext(output_file)[0] + ".point"
    with open(point_file, 'w') as f:
        for x, y in coordinates:
            f.write(f"{x} {y} 0\n")  # Writing each coordinate as a new line in the .point file

    # Step 2: Convert the .point file to a .mod file
    convert_point_to_model(point_file, output_file)

def write_coord_file(coordinates, output_file):
    """
    Write a generic .coord file with particle coordinates.

    :param list_of_tuples coordinates: List of (x, y) coordinates for the particles.
    :param str output_file: Output file path for the .coord file.
    """
    print_and_log("", logging.DEBUG)
    coord_file = os.path.splitext(output_file)[0] + ".coord"
    with open(coord_file, 'w') as f:
        for x, y in coordinates:
            f.write(f"{x} {y}\n")  # Writing each coordinate as a new line in the .coord file

def save_particle_coordinates(structure_name, particle_locations, output_path, imod_coordinate_file, coord_coordinate_file):
    """
    Saves particle coordinates in specified formats (.star, .mod, .coord).

    :param str structure_name: Base name for the output files.
    :param list particle_locations: List of particle locations as (x, y) tuples.
    :param Namespace args: Command-line arguments containing user preferences for file outputs.
    :param str prefix: Optional prefix for file names.
    :param str suffix: Optional suffix for file names, e.g., for handling repeats.
    """
    print_and_log("", logging.DEBUG)
    # Save .star file
    write_all_coordinates_to_star(structure_name, output_path + ".mrc", particle_locations)

    # Save IMOD .mod files
    if imod_coordinate_file:
        write_mod_file(particle_locations, os.path.splitext(output_path)[0] + ".mod")
        #write_mod_file(removed_particles, os.path.splitext(point_file_path)[0] + "_removed.mod")  # Writes the particles that were removed

    # Save .coord files
    if coord_coordinate_file:
        write_coord_file(particle_locations, os.path.splitext(output_path)[0] + ".coord")


def estimate_mass_from_map(mrc_name):
    """
    Estimate the mass of a protein in a cryoEM density map.

    :param str mrc_path: Path to the MRC/MAP file.
    :return float: Estimated mass of the protein in kilodaltons (kDa).

    This function estimates the mass of a protein based on the volume of density present in a cryoEM density map (MRC/MAP file) and the provided pixel size. It assumes an average protein density of 1.35 g/cm³ and uses the volume of voxels above a certain threshold to represent the protein. The threshold is set as the mean plus one standard deviation of the density values in the map. This is a simplistic thresholding approach and might need adjustment based on the specific map and protein.

    The estimated mass is returned in kilodaltons (kDa), considering the conversion from grams to daltons and then to kilodaltons.

    Note: This method assumes the map is already thresholded appropriately and that the entire volume above the threshold corresponds to protein. In practice, determining an effective threshold can be challenging and may require manual intervention or advanced image analysis techniques.
    """
    print_and_log("", logging.DEBUG)
    protein_density_g_per_cm3 = 1.35  # Average density of protein
    angstroms_cubed_to_cm_cubed = 1e-24  # Conversion factor

    mrc_path = f"{mrc_name}.mrc"
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data
        pixel_size_angstroms = mrc.voxel_size.x  # Assuming cubic voxels; adjust if necessary
        # Example threshold; customize as needed
        threshold = data.mean() + data.std()
        voxel_volume_angstroms_cubed = pixel_size_angstroms**3
        protein_volume_angstroms_cubed = np.sum(data > threshold) * voxel_volume_angstroms_cubed
        protein_volume_cm_cubed = protein_volume_angstroms_cubed * angstroms_cubed_to_cm_cubed

    mass_g = protein_volume_cm_cubed * protein_density_g_per_cm3
    mass_daltons = mass_g / 1.66053906660e-24  # Convert grams to daltons
    mass_kDa = mass_daltons / 1000  # Convert daltons to kilodaltons

    return mass_kDa

def get_mrc_box_size(mrc_file_path):
    """
    Reads an MRC file and returns its box size, assuming the file is a cube.

    :param str mrc_file_path: The file path to the MRC file whose box size is to be determined.
    :return int: The size of one side of the cubic box in pixels.
    :raises FileNotFoundError: If the specified MRC file does not exist.
    :raises Exception: If there are issues reading the MRC file, indicating it might be corrupted or improperly formatted.
    """
    print_and_log("", logging.DEBUG)
    with mrcfile.open(mrc_file_path, permissive=True) as mrc:
        box_size = mrc.data.shape[0]  # Assuming the map is a cube
    return box_size

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

def extend_and_shuffle_image_list(num_images, image_list_file):
    """
    Extend (if necessary), shuffle, and select a specified number of random ice micrographs.

    :param int num_images: The number of images to select.
    :param str image_list_file: The path to the file containing the list of images.
    :return list: A list of selected ice micrograph filenames and defoci.
    """
    print_and_log("", logging.DEBUG)
    print_and_log(f"Selecting {num_images} random ice micrographs...", logging.INFO)
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

    print_and_log("Done!\n", logging.INFO)

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

def fourier_crop(image, downsample_factor):
    """	
    Fourier crops a 2D image.

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

def downsample_micrograph(image_path, downsample_factor):
    """
    Downsample a micrograph by Fourier cropping and save it to a temporary directory.
    Supports mrc, png, and jpeg formats.

    :param str image_path: Path to the micrograph image file.
    :param int downsample_factor: Factor by which to downsample the image in both dimensions.
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
        downsampled_image = fourier_crop(image, downsample_factor)

        # Create a bin directory to store the downsampled micrograph
        bin_dir = os.path.join(os.path.dirname(image_path), f"bin_{downsample_factor}")
        os.makedirs(bin_dir, exist_ok=True)

        # Save the downsampled micrograph with the same name plus _bin## in the binned directory
        binned_image_path = os.path.join(bin_dir, f"{name}_bin{downsample_factor}{ext}")
        if ext == '.mrc':
            writemrc(binned_image_path, ((downsampled_image - np.mean(downsampled_image)) / np.std(downsampled_image)))  # Save image with 0 mean and std of 1
        elif ext in ['.png', '.jpeg']:
            # Normalize image to [0, 255] and convert to uint8
            downsampled_image -= downsampled_image.min()
            downsampled_image = downsampled_image / downsampled_image.max() * 255.0
            downsampled_image = downsampled_image.astype(np.uint8)
            cv2.imwrite(binned_image_path, downsampled_image)

    except Exception as e:
        print_and_log(f"Error processing {image_path}: {str(e)}", logging.INFO)

def parallel_downsample(image_directory, cpus, downsample_factor):
    """
    Downsample all micrographs in a directory in parallel.

    :param str image_directory: Local directory name where the micrographs are stored in mrc, png, and/or jpeg formats.
    :param int downsample_factor: Factor by which to downsample the images in both dimensions.
    """
    print_and_log("", logging.DEBUG)
    image_extensions = ['.mrc', '.png', '.jpeg']
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if os.path.splitext(filename)[1].lower() in image_extensions]

    # Create a pool of worker processes
    pool = Pool(processes=cpus)

    # Downsample each micrograph by processing each image path in parallel
    pool.starmap(downsample_micrograph, zip(image_paths, itertools.repeat(downsample_factor)))

    # Close the pool to prevent any more tasks from being submitted
    pool.close()

    # Wait for all worker processes to finish
    pool.join()

def downsample_star_file(input_star, output_star, downsample_factor):
    """
    Read a STAR file, downsample the coordinates, and write a new STAR file.

    :param str input_star: Path to the input STAR file.
    :param str output_star: Path to the output STAR file.
    :param int downsample_factor: Factor by which to downsample the coordinates.
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
    :param bool coord_coordinate_file: Whether to downsample and save generic .coord coordinate files.
    """
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
    :return dataframe: DataFrame with columns for micrograph names, particle coordinates, angles, and optics group.
    :raises ValueError: If data_particles section of the STAR file is not found.
    """
    print_and_log("", logging.DEBUG)
    # Track the line number for where data begins
    data_start_line = None
    with open(star_file_path, 'r') as file:
        for i, line in enumerate(file):
            if 'data_particles' in line:
                # Found the data_particles section, now look for the actual data start
                data_start_line = i + 8  # Adjust if more lines are added to the star file
                break

    if data_start_line is None:
        raise ValueError("data_particles section not found in the STAR file.")

    # Read the data section from the identified start line, adjusting for the actual data start
    # Correct the `skiprows` approach to accurately target the start of data rows
    # Use `comment='#'` to ignore lines starting with '#'
    df = pd.read_csv(star_file_path, sep='\s+', skiprows=lambda x: x < data_start_line, header=None,
                     names=['micrograph_name', 'coord_x', 'coord_y', 'angle', 'optics_group'], comment='#')

    return df

def trim_vol_return_rand_particle_number(input_mrc, input_micrograph, scale_percent, output_mrc):
    """
    Trim a volume and return a random number of particles within a micrograph based on the maximum
    number of projections of this volume that can fit in the micrograph without overlapping.

    :param str input_mrc: The file path of the input volume in MRC format.
    :param str input_micrograph: The file path of the input micrograph in MRC format.
    :param float scale_percent: The percentage to scale the volume for trimming.
    :param str output_mrc: The file path to save the trimmed volume.
    :return int: A random number of particles up to a maximum of how many will fit in the micrograph.
    """
    print_and_log("", logging.DEBUG)
    mrc_array = readmrc(input_mrc)
    micrograph_array = readmrc(input_micrograph)

    # Find the non-zero entries and their indices
    non_zero_indices = np.argwhere(mrc_array)

    # Find the minimum and maximum indices for each dimension
    min_indices = np.min(non_zero_indices, axis=0)
    max_indices = np.max(non_zero_indices, axis=0) + 1

    # Compute the size of the largest possible equilateral cube
    cube_size = np.max(max_indices - min_indices)

    # Increase the cube size by #%
    cube_size = int(np.ceil(cube_size * (100 + scale_percent)/100))

    # Find the next largest number that is divisible by at least 3 of the 5 smallest prime numbers
    primes = [2, 3, 5]
    cube_size = min(next_divisible_by_primes(cube_size, primes, 2), 336)  # 320 is the largest practical box size before memory issues or seg faults

    # Adjust the minimum and maximum indices to fit the equilateral cube
    min_indices -= (cube_size - (max_indices - min_indices)) // 2
    max_indices = min_indices + cube_size

    # Handle boundary cases to avoid going beyond the original array size
    min_indices = np.maximum(min_indices, 0)
    max_indices = np.minimum(max_indices, mrc_array.shape)

    # Slice the original array to obtain the trimmed array
    trimmed_mrc_array = mrc_array[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1], min_indices[2]:max_indices[2]]

    writemrc(output_mrc, trimmed_mrc_array)

    # Set the maximum number of particle projections that can fit in the image
    max_num_particles = int(2*micrograph_array.shape[0]*micrograph_array.shape[1]/(trimmed_mrc_array.shape[0]*trimmed_mrc_array.shape[1]))

    # Choose a random number of particles between 2 and max, with low particle numbers (<100) downweighted
    rand_num_particles = non_uniform_random_number(2, max_num_particles, 100, 0.5)

    return rand_num_particles, max_num_particles

def filter_coordinates_outside_polygons(particle_locations, json_scale, polygons):
    """
    Filters out particle locations that are inside any polygon.

    :param list_of_tuples particle_locations: List of (x, y) coordinates of particle locations.
    :param int json_scale: Binning factor used when labeling junk to create the json file.
    :param list_of_tuples polygons: List of polygons where each polygon is a list of (x, y) coordinates.
    :return list_of_tuples : List of (x, y) coordinates of particle locations that are outside the polygons.
    """
    print_and_log("", logging.DEBUG)
    # An empty list to store particle locations that are outside the polygons
    filtered_particle_locations = []

    # Scale particle locations us to the proper image size
    particle_locations = [(float(x)/json_scale, float(y)/json_scale) for x, y in particle_locations]

    # Iterate over each particle location
    for x, y in particle_locations:
        # Variable to keep track if a point is inside any polygon
        inside_any_polygon = False

        # Check each polygon to see if the point is inside
        for polygon in polygons:
            path = Path(polygon)
            if path.contains_point((x, y)):
                inside_any_polygon = True
                break  # Exit the loop if point is inside any polygon

        # If the point is not inside any polygon, add it to the filtered list
        if not inside_any_polygon:
            filtered_particle_locations.append((x, y))

    # Scale filtered particle locations back up
    filtered_particle_locations = [(float(x) * json_scale, float(y) * json_scale) for x, y in filtered_particle_locations]

    return filtered_particle_locations

def generate_particle_locations(micrograph_image, image_size, num_small_images, half_small_image_width, border_distance, edge_particles, dist_type, non_random_dist_type):
    """
    Generate random/non-random locations for particles within an image.

    :param numpy_array micrograph_image: The micrograph image (used only in the 'micrograph' distribution option.
    :param tuple image_size: The size of the image as a tuple (width, height).
    :param int num_small_images: The number of small images or particles to generate coordinates for.
    :param int half_small_image_width: Half the width of a small image.
    :param int border_distance: The minimum distance between particles and the image border.
    :param bool edge_particles: Allow particles to be placed up to the edge of the micrograph.
    :param str dist_type: Particle location generation distribution type - 'random' or 'non-random'.
    :param str non_random_dist_type: Type of non-random distribution when dist_type is 'non-random' - 'circular', 'inverse circular', or 'gaussian'.
    :return list_of_tuples: A list of particle locations as tuples (x, y).
    """
    print_and_log("", logging.DEBUG)
    width, height = image_size

    # If edge_particles is set to True, allow particles to go all the way to the edge of the micrograph
    border_distance = -1 if edge_particles else max(border_distance, half_small_image_width)

    particle_locations = []

    def is_far_enough(new_particle_location, particle_locations, half_small_image_width):
        """
        Check if a new particle location is far enough from existing particle locations.

        :param tuple new_particle_location: The new particle location as a tuple (x, y).
        :param tuple particle_locations: The existing particle locations.
        :param int half_small_image_width: Half the width of a small image.
        :return bool: True if the new particle location is far enough, False otherwise.
        """
        for particle_location in particle_locations:
            distance = np.sqrt((new_particle_location[0] - particle_location[0])**2 + (new_particle_location[1] - particle_location[1])**2)
            if distance < half_small_image_width:
                return False
        return True

    max_attempts = 1000  # Maximum number of attempts to find an unoccupied point in the distribution

    if dist_type == 'random':
        attempts = 0  # Counter for attempts to find a valid position
        # Keep generating and appending particle locations until we have enough.
        while len(particle_locations) < num_small_images and attempts < max_attempts:
            x = np.random.randint(border_distance, width - border_distance)
            y = np.random.randint(border_distance, height - border_distance)
            new_particle_location = (x, y)
            if is_far_enough(new_particle_location, particle_locations, half_small_image_width):
                particle_locations.append(new_particle_location)
                attempts = 0  # Reset attempts counter after successful addition
            else:
                attempts += 1  # Increment attempts counter if addition is unsuccessful

    elif dist_type == 'non-random':
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
                if is_far_enough(new_particle_location, particle_locations, half_small_image_width) and border_distance <= x <= width - border_distance and border_distance <= y <= height - border_distance:
                    particle_locations.append(new_particle_location)
                    attempts = 0  # Reset attempts counter after successful addition
                else:
                    attempts += 1  # Increment attempts counter if addition is unsuccessful

        elif non_random_dist_type == 'inverse circular':
            # Parameters for the exclusion zone
            # Randomly determine the center within the image, away from the edges
            exclusion_center_x = np.random.randint(border_distance + half_small_image_width, width - border_distance - half_small_image_width)
            exclusion_center_y = np.random.randint(border_distance + half_small_image_width, height - border_distance - half_small_image_width)
            exclusion_center = (exclusion_center_x, exclusion_center_y)

            # Determine the maximum possible radius for the exclusion zone based on the image size and center position
            max_radius = min(width // 2, height // 2)

            # Randomly select a radius for the exclusion zone
            exclusion_radius = np.random.randint(half_small_image_width, max_radius)

            # Generate particle locations avoiding the central circle
            attempts = 0  # Counter for attempts to find a valid position outside the exclusion zone
            while len(particle_locations) < num_small_images and attempts < max_attempts:
                x = np.random.randint(border_distance, width - border_distance)
                y = np.random.randint(border_distance, height - border_distance)

                # Check if the location is outside the exclusion zone
                if np.sqrt((x - exclusion_center_x) ** 2 + (y - exclusion_center_y) ** 2) > exclusion_radius:
                    if is_far_enough((x, y), particle_locations, half_small_image_width):
                        particle_locations.append((x, y))
                        attempts = 0  # Reset attempts counter after successful addition
                    else:
                        attempts += 1  # Increment attempts counter if addition is unsuccessful
                else:
                    attempts += 1  # Increment attempts counter if location is inside the exclusion zone

        elif non_random_dist_type == 'gaussian':
            num_gaussians = np.random.randint(1, 6)  # Random number of Gaussian distributions between 1 and 5
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

            attempts = 0  # Reset attempts counter for Gaussian distribution
            # Keep generating and appending particle locations until we have enough.
            while len(particle_locations) < num_small_images and attempts < max_attempts:
                # Randomly select one of the Gaussian distributions.
                chosen_gaussian = np.random.choice(num_gaussians)
                center, stddev = gaussians[chosen_gaussian]
                # Generate random x and y coordinates from the chosen Gaussian.
                x = int(np.random.normal(center[0], stddev))
                y = int(np.random.normal(center[1], stddev))
                new_particle_location = (x, y)
                if border_distance <= x <= width - border_distance and border_distance <= y <= height - border_distance and is_far_enough(new_particle_location, particle_locations, half_small_image_width):
                    particle_locations.append(new_particle_location)
                    attempts = 0  # Reset attempts counter after successful addition
                else:
                    attempts += 1  # Increment attempts counter if addition is unsuccessful

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
            idx = np.arange(width * height)

            # Choose locations based on the probability map
            chosen_indices = np.random.choice(idx, size=num_small_images, replace=False, p=flat_prob_map)
            y_coords, x_coords = np.unravel_index(chosen_indices, (height, width))

            # Ensure particles are placed within borders
            for x, y in zip(x_coords, y_coords):
                if border_distance <= x <= width - border_distance and border_distance <= y <= height - border_distance:
                    particle_locations.append((x, y))

    return particle_locations

def estimate_noise_parameters(mrc_path):
    """
    Estimate Poisson (shot noise) and Gaussian (readout and electronic noise) parameters from a single MRC image.

    :param mrc_path: Path to the MRC file.
    :return float, float: A tuple containing the estimated Poisson variance (mean) and Gaussian variance.
    """
    with mrcfile.open(mrc_path, mode='r') as mrc:
        image = mrc.data.astype(np.float32)
        
        # Calculate mean and variance across the image
        mean = np.mean(image)
        variance = np.var(image)
        
        # Poisson noise component is approximated by the mean
        # Gaussian noise component is the excess variance over the Poisson component
        gaussian_variance = variance - mean
        
        return mean, gaussian_variance

def process_slice(args):
    """
    Process a slice of the particle stack by adding Poisson noise.

    :param args: A tuple containing the following parameters:
                 - slice numpy_array: A 3D numpy array representing a slice of the particle stack.
                 - num_frames int: Number of frames to simulate for each particle image.
                 - scaling_factor float: Factor by which to scale the particle images before adding noise.
    :return numpy_array: A 3D numpy array representing the processed slice of the particle stack with added noise.
    """
    print_and_log("", logging.DEBUG)
    # Unpack the arguments
    slice, num_frames, scaling_factor = args

    # Create an empty array to store the noisy slice of the particle stack
    noisy_slice = np.zeros_like(slice, dtype=np.float32)

    # Iterate over each particle in the slice
    for i in range(slice.shape[0]):
        # Get the i-th particle and scale it by the scaling factor
        particle = slice[i, :, :] * scaling_factor

        # Create a mask for non-zero values in the particle
        mask = particle > 0

        # For each frame, simulate the noise and accumulate the result
        for _ in range(num_frames):
            # Initialize a frame with zeros
            noisy_frame = np.zeros_like(particle, dtype=np.float32)

            # Add Poisson noise to the non-zero values in the particle, modulated by the original pixel values; it represents shot noise.
            noisy_frame[mask] = np.random.poisson(particle[mask])

            # Accumulate the noisy frame to the noisy slice
            noisy_slice[i, :, :] += noisy_frame

    return noisy_slice

def add_poisson_noise(particle_stack, num_frames, num_cores, scaling_factor=1.0):
    """
    Add Poisson noise to a stack of particle images.

    This function simulates the acquisition of `num_frames` frames for each particle image
    in the input stack, adds Poisson noise to each frame, and then sums up the frames to
    obtain the final noisy particle image. The function applies both noises only to the
    non-zero values in each particle image, preserving the background.

    :param numpy_array particle_stack: 3D numpy array representing a stack of 2D particle images.
    :param int num_frames: Number of frames to simulate for each particle image.
    :param int num_cores: Number of CPU cores to parallelize slices across.
    :param float scaling_factor: Factor by which to scale the particle images before adding noise.
    :return numpy_array: 3D numpy array representing the stack of noisy particle images.
    """
    print_and_log("", logging.DEBUG)
    # Split the particle stack into slices
    slices = np.array_split(particle_stack, num_cores)

    # Prepare the arguments for each slice
    args = [(s, num_frames, scaling_factor) for s in slices]

    # Create a pool of worker processes
    with Pool(num_cores) as pool:
        # Process each slice in parallel
        noisy_slices = pool.map(process_slice, args)

    # Concatenate the processed slices back into a single stack
    noisy_particle_stack = np.concatenate(noisy_slices, axis=0)

    return noisy_particle_stack

def create_collage(large_image, small_images, particle_locations, gaussian_variance):
    """
    Create a collage of small images on a blank canvas of the same size as the large image.
    Add Gaussian noise based on the micrograph that the particle collage will be projected onto.
    Particles that would extend past the edge of the large image are trimmed before being added.

    :param numpy_array large_image: Shape of the large image.
    :param numpy_array small_images: List of small images to place on the canvas.
    :param list_of_tuples particle_locations: Coordinates where each small image should be placed.
    :param float gaussian_variance: Standard deviation of the Gaussian electronic noise, as measured previously from the ice image.
    :return numpy_array: Collage of small images.
    """
    print_and_log("", logging.DEBUG)
    collage = np.zeros(large_image.shape, dtype=large_image.dtype)

    for i, small_image in enumerate(small_images):
        x, y = particle_locations[i]
        x_start = x - small_image.shape[0] // 2
        y_start = y - small_image.shape[1] // 2

        x_end = x_start + small_image.shape[0]
        y_end = y_start + small_image.shape[1]

        # Calculate the region of the small image that fits within the large image
        x_start_trim = max(0, -x_start)
        y_start_trim = max(0, -y_start)
        x_end_trim = min(small_image.shape[0], large_image.shape[1] - x_start)
        y_end_trim = min(small_image.shape[1], large_image.shape[0] - y_start)

        # Adjust start and end coordinates to ensure they fall within the large image
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(large_image.shape[1], x_end)
        y_end = min(large_image.shape[0], y_end)

        # Before adding, ensure the sizes match by explicitly setting the shape dimensions
        trim_height, trim_width = y_end_trim - y_start_trim, x_end_trim - x_start_trim
        y_end = y_start + trim_height
        x_end = x_start + trim_width

        # Add the trimmed small image to the collage
        collage[y_start:y_end, x_start:x_end] += small_image[y_start_trim:y_end_trim, x_start_trim:x_end_trim]

    # Apply Gaussian noise across the entire collage to simulate the camera noise
    gaussian_noise = np.random.normal(loc=0, scale=np.sqrt(gaussian_variance), size=collage.shape)
    collage += gaussian_noise

    return collage

def blend_images(large_image, small_images, scale_percent, half_small_image_width, particle_locations, border_distance, save_edge_coordinates, scale, structure_name, imod_coordinate_file, coord_coordinate_file, large_image_path, output_path, no_junk_filter, json_scale, flip_x, flip_y, polygon_expansion_distance, gaussian_variance):
    """
    Blend small images (particles) into a large image (micrograph).
    Also makes coordinate files.

    :param numpy_array large_image: The large image where small images will be blended into.
    :param numpy_array small_images: The list of small images or particles to be blended.
    :param float scale_percent: The percentage to scale the volume for trimming.
    :param int half_small_image_width: Half the width of a small image.
    :param list_of_tuples particle_locations: The locations of the particles within the large image.
    :param int border_distance: The minimum distance between particles and the image border.
    :param bool save_edge_coordinates: Save particle coordinates that are closer than half a particle box size from the edge.
    :param float scale: The scale factor to adjust the intensity of the particles.
    :param str structure_name: The name of the structure file.
    :param bool imod_coordinate_file: Boolean for whether or not to output an IMOD .mod coordinate file.
    :param bool coord_coordinate_file: Boolean for whether or not to output a generic .coord coordinate file.
    :param str large_image_path: The path of the micrograph.
    :param str image_path: The output path.
    :param bool no_junk_filter: Boolean for whether or not to filter junk from coordinate file locations.
    :param int json_scale: Binning factor used when labeling junk to create the json file.
    :param bool flip_x: Boolean to determine if the x-coordinates should be flipped.
    :param bool flip_y: Boolean to determine if the y-coordinates should be flipped.
    :param int polygon_expansion_distance: Distance by which to expand the polygons.
    :param float gaussian_variance: Standard deviation of the Gaussian electronic noise, as measured previously from the ice image.
    :return numpy_array: The blended large image.
    """
    print_and_log("", logging.DEBUG)
    json_file_path = os.path.splitext(large_image_path)[0] + ".json"
    if not no_junk_filter:
        if os.path.exists(json_file_path):
            polygons = read_polygons_from_json(json_file_path, polygon_expansion_distance, flip_x, flip_y, expand=True)

            # Remove particle locations from inside polygons (junk in micrographs) when writing coordinate files
            filtered_particle_locations = filter_coordinates_outside_polygons(particle_locations, json_scale, polygons)
            num_particles_removed = len(particle_locations) - len(filtered_particle_locations)
            print_and_log(f"{num_particles_removed} particles removed from coordinate file(s) based on the corresponding JSON file.", logging.INFO)
        else:
            print_and_log(f"JSON file with polygons for bad micrograph areas not found: {json_file_path}", logging.WARNING)
            filtered_particle_locations = particle_locations
    else:
        print_and_log("Skipping junk filtering (ie. not using JSON file)", logging.INFO)
        filtered_particle_locations = particle_locations

    # Remove edge particles from coordinate files if requested
    if not save_edge_coordinates:
        remaining_particle_locations = filtered_particle_locations[:]

        # Loop through each particle location to check its proximity to the edge
        for x, y in filtered_particle_locations:
            # Calculate the effective borders for this particle
            reduced_sidelength = int(np.ceil(half_small_image_width * 100/(100 + scale_percent)))  # Unscale the sidelength, which was previously scaled up to capture more CTF
            left_edge = x - reduced_sidelength
            right_edge = x + reduced_sidelength
            top_edge = y - reduced_sidelength
            bottom_edge = y + reduced_sidelength

            # Determine if the particle is too close to any edge of the large image
            if (left_edge < border_distance or right_edge > large_image.shape[1] - border_distance or
                top_edge < border_distance or bottom_edge > large_image.shape[0] - border_distance):
                # Remove this particle's location if we're not saving edge coordinates
                if not save_edge_coordinates:
                    remaining_particle_locations.remove((x, y))

        num_particles_removed = len(filtered_particle_locations) - len(remaining_particle_locations)
        if num_particles_removed > 0:
            print_and_log(f"{num_particles_removed} particles removed from coordinate file(s) due to being too close to the edge.", logging.INFO)
        else:
            print_and_log("No particles removed from coordinate file(s) due to being too close to the edge.", logging.INFO)

        # Use the remaining locations for further processing
        filtered_particle_locations = remaining_particle_locations

    # Normalize the input micrograph to itself
    large_image[:, :] = (large_image[:, :] - large_image[:, :].mean())/large_image[:, :].std()

    collage = create_collage(large_image, small_images, particle_locations, gaussian_variance)
    collage *= scale  # Apply scaling if necessary

    blended_image = large_image + collage  # Blend the collage with the large image

    # Normalize the resulting micrograph to itself
    blended_image = (blended_image - blended_image.mean()) / blended_image.std()

    save_particle_coordinates(structure_name, filtered_particle_locations, output_path, imod_coordinate_file, coord_coordinate_file)

    return blended_image, filtered_particle_locations

def add_images(large_image_path, small_images, scale_percent, structure_name, border_distance, edge_particles, save_edge_coordinates, scale, output_path, dist_type, non_random_dist_type, imod_coordinate_file, coord_coordinate_file, no_junk_filter, json_scale, flip_x, flip_y, polygon_expansion_distance, gaussian_variance, save_as_mrc, save_as_png, save_as_jpeg, jpeg_quality):
    """
    Add small images or particles to a large image and save the resulting micrograph.

    :param str large_image_path: The file path of the large image or micrograph.
    :param str small_images: The file path of the small images or particles.
    :param float scale_percent: The percentage to scale the volume for trimming.
    :param str structure_name: The name of the structure file.
    :param int border_distance: The minimum distance between particles and the image border.
    :param bool edge_particles: Allow particles to be placed up to the edge of the micrograph.
    :param bool save_edge_coordinates: Save particle coordinates that are closer than half a particle box size from the edge.
    :param float scale: The scale factor to adjust the intensity of the particles. Adjusted based on ice_thickness parameters.
    :param str output_path: The file path to save the resulting micrograph.
    :param str dist_type: The type of distribution (random or non-random) for placing particles in micrographs.
    :param str non_random_dist_type: The type of non-random distribution (circular, inverse circular, gaussian) for placing particles in micrographs.
    :param bool imod_coordinate_file: Boolean for whether or not to output an IMOD .mod coordinate file.
    :param bool coord_coordinate_file: Boolean for whether or not to output a generic .coord coordinate file.
    :param bool no_junk_filter: Boolean for whether or not to filter junk from coordinate file locations.
    :param float json_scale: Binning factor used when labeling junk to create the json file.
    :param bool flip_x: Boolean to determine if the x-coordinates should be flipped.
    :param bool flip_y: Boolean to determine if the y-coordinates should be flipped.
    :param float polygon_expansion_distance: Number of pixels to expand each polygon in the json file that defines areas to not place particle coordinates.
    :param float gaussian_variance: Standard deviation of the Gaussian electronic noise, as measured previously from the ice image.
    :param bool save_as_mrc: Boolean to save the resulting synthetic micrograph and an MRC file.
    :param bool save_as_png: Boolean to save the resulting synthetic micrograph and an PNG file.
    :param bool save_as_jpeg: Boolean to save the resulting synthetic micrograph and an JPEG file.
    :param int jpeg_quality: Quality of the JPEG image.
    :return int: The actual number of particles added to the micrograph.
    """
    print_and_log("", logging.DEBUG)
    # Read micrograph and particles, and get some information
    large_image = readmrc(large_image_path)
    small_images = readmrc(small_images)
    image_size = np.flip(large_image.shape)
    num_small_images = len(small_images)
    half_small_image_width = int(small_images.shape[1]/2)

    # Generates unfiltered particle locations, which may be filtered of junk and/or edge particles in blend_images
    particle_locations = generate_particle_locations(large_image, image_size, num_small_images, half_small_image_width, border_distance, edge_particles, dist_type, non_random_dist_type)

    # Blend the images together
    if len(particle_locations) == num_small_images:
        result_image, filtered_particle_locations = blend_images(large_image, small_images, scale_percent, half_small_image_width, particle_locations, border_distance, save_edge_coordinates, scale, structure_name, imod_coordinate_file, coord_coordinate_file, large_image_path, output_path, no_junk_filter, json_scale, flip_x, flip_y, polygon_expansion_distance, gaussian_variance)
    else:
        print_and_log(f"Only {len(particle_locations)} could fit into the image. Adding those to the micrograph now...", logging.INFO)
        result_image, filtered_particle_locations = blend_images(large_image, small_images[:len(particle_locations), :, :], scale_percent, half_small_image_width, particle_locations, border_distance, save_edge_coordinates, scale, structure_name, imod_coordinate_file, coord_coordinate_file, large_image_path, output_path, no_junk_filter, json_scale, flip_x, flip_y, polygon_expansion_distance, gaussian_variance)

    # Save the resulting micrograph in specified formats
    if save_as_mrc:
        print_and_log(f"\nWriting synthetic micrograph as a MRC file: {output_path}.mrc...\n", logging.INFO)
        writemrc(output_path + '.mrc', (result_image - np.mean(result_image)) / np.std(result_image))  # Write mrc normalized with mean of 0 and std of 1
    if save_as_png:
        # Needs to be scaled from 0 to 255 and flipped
        result_image -= result_image.min()
        result_image = result_image / result_image.max() * 255.0
        print_and_log(f"\nWriting synthetic micrograph as a PNG file: {output_path}.png...\n", logging.INFO)
        cv2.imwrite(output_path + '.png', np.flip(result_image, axis=0))
    if save_as_jpeg:
        # Needs to be scaled from 0 to 255 and flipped
        result_image -= result_image.min()
        result_image = result_image / result_image.max() * 255.0
        print_and_log(f"\nWriting synthetic micrograph as a JPEG file: {output_path}.jpeg...\n", logging.INFO)
        cv2.imwrite(output_path + '.jpeg', np.flip(result_image, axis=0), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    return len(particle_locations), len(filtered_particle_locations)

def crop_particles(micrograph_path, particle_rows, particles_dir, box_size):
    """
    Crops particles from a single micrograph.
    
    :param str micrograph_path: Path to the micrograph.
    :param DataFrame particle_rows: DataFrame rows of particles to be cropped from the micrograph.
    :param str particles_dir: Directory to save cropped particles.
    :param int box_size: The box size in pixels for the cropped particles.
    """
    with mrcfile.open(micrograph_path, permissive=True) as mrc:
        for _, row in particle_rows.iterrows():
            x, y = int(row['coord_x']), int(row['coord_y'])
            half_box_size = box_size // 2

            if x - half_box_size < 0 or y - half_box_size < 0 or x + half_box_size > mrc.data.shape[1] or y + half_box_size > mrc.data.shape[0]:
                continue

            cropped_particle = mrc.data[y-half_box_size:y+half_box_size, x-half_box_size:x+half_box_size]
            particle_path = os.path.join(particles_dir, f"particle_{row['particle_counter']:010d}.mrc")
            writemrc(particle_path, cropped_particle.astype(np.float32))

def crop_particles_from_micrographs(structure_dir, box_size, num_cpus):
    """
    Crops particles from micrographs based on coordinates specified in a micrograph STAR file
    and saves them with a specified box size. This function operates in parallel, using a specified
    number of CPU cores to process different micrographs concurrently.

    :param str structure_dir: The directory containing the structure's micrographs and STAR file.
    :param int box_size: The box size in pixels for the cropped particles. If None, the function
                         will dynamically determine the box size from the .mrc map used for projections.
    :param int num_cpus: The number of CPU cores to use for parallel processing. If not specified,
                         all available cores are used.

    Notes:
    - Particles whose specified box would extend beyond the micrograph boundaries are not cropped.
    - This function assumes the presence of a STAR file in the structure directory with the naming
      convention '{structure_name}.star' containing the necessary coordinates for cropping.
    """
    particles_dir = os.path.join(structure_dir, 'Particles/')
    os.makedirs(particles_dir, exist_ok=True)

    star_file_path = os.path.join(structure_dir, f'{structure_dir}.star')
    df = read_star_particles(star_file_path)
    df['particle_counter'] = range(1, len(df) + 1)
    
    grouped_df = df.groupby('micrograph_name')

    # Use the user-defined number of CPUs for parallel processing
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = []
        for micrograph_name, particle_rows in grouped_df:
            micrograph_path = os.path.join(micrograph_name)
            if not os.path.exists(micrograph_path):
                print_and_log(f"Micrograph not found: {micrograph_path}", logging.WARNING)
                continue
            
            print_and_log(f"Extracting {len(particle_rows)} particles for {micrograph_name}...", logging.INFO)
            future = executor.submit(crop_particles, micrograph_path, particle_rows, particles_dir, box_size)
            futures.append(future)
        
        # Optional: Wait for all futures to complete if you need to process results further
        for future in futures:
            future.result()

def generate_micrographs(args, structure_name, structure_type, structure_index, total_structures):
    """
    Generate synthetic micrographs for a specified structure.

    This function orchestrates the workflow for generating synthetic micrographs
    for a given structure. It performs file download, conversion, image shuffling,
    and iterates through the generation process for each selected image. It also
    handles cleanup operations post-generation.

    :param str structure_name: The structure name from which synthetic micrographs are to be generated.
    :param Namespace args: The argument namespace containing all the command-line arguments specified by the user.

    :return int int: The total number of particles actually added to all of the micrographs, and the box size of the projected volume.
    """
    print_and_log("", logging.DEBUG)
    # Convert short distribution to full distribution name
    distribution_mapping = {'r': 'random', 'n': 'non-random'}
    distribution = distribution_mapping.get(args.distribution, args.distribution)

    # Create output directory
    if not os.path.exists(structure_name):
        os.mkdir(structure_name)

    # Convert PDB to MRC for PDBs
    if structure_type == "pdb":
        mass = convert_pdb_to_mrc(structure_name, args.apix, args.pdb_to_mrc_resolution)
        print_and_log(f"Mass of PDB {structure_name}: {mass} kDa", logging.INFO)
        fudge_factor = 5
    elif structure_type == "mrc":
        mass = int(estimate_mass_from_map(structure_name))
        print_and_log(f"Estimated mass of MRC {structure_name}: {mass} kDa", logging.INFO)
        fudge_factor = 4

    # Write STAR header for the current synthetic dataset
    write_star_header(structure_name, args.apix, args.voltage, args.Cs)

    # Shuffle and possibly extend the ice images
    selected_images = extend_and_shuffle_image_list(args.num_images, args.image_list_file)

    # Main Loop
    total_num_particles_projected = 0
    total_num_particles_with_saved_coordinates = 0
    current_micrograph_number = 0
    micrograph_usage_count = {}  # Dictionary to keep track of repeating micrograph names if the image list was extended
    for line in selected_images:
        current_micrograph_number += 1
        # Parse the 'micrograph_name.mrc defocus' line
        fields = line.strip().split()
        fname, defocus = fields[0], fields[1]
        # Update micrograph usage count
        if fname in micrograph_usage_count:
            micrograph_usage_count[fname] += 1
        else:
            micrograph_usage_count[fname] = 1
        # Generate the repeat number suffix for the filename
        repeat_number = micrograph_usage_count[fname]
        repeat_suffix = f"{repeat_number}" if repeat_number > 1 else ""

        extra_hyphens = '-' * (len(str(current_micrograph_number)) + len(str(args.num_images)) + len(str(structure_name)) + len(str(structure_index)) + len(str(total_structures)) + len(str(fname)))
        print_and_log(f"\n-------------------------------------------------------{extra_hyphens}", logging.WARNING)
        print_and_log(f"Generating synthetic micrograph ({current_micrograph_number}/{args.num_images}) using {structure_name} ({structure_index + 1}/{total_structures}) from {fname}...", logging.WARNING)
        print_and_log(f"-------------------------------------------------------{extra_hyphens}\n", logging.WARNING)

        # Random or user-specified ice thickness
        # Adjust the relative ice thickness to work mathematically (yes, the inputs are inversely named and there is a fudge factor of 5 just so the user gets a number that feels right)
        if args.ice_thickness is None:
            min_ice_thickness = fudge_factor/args.max_ice_thickness
            max_ice_thickness = fudge_factor/args.min_ice_thickness
            ice_thickness = random.uniform(min_ice_thickness, max_ice_thickness)
        else:
            ice_thickness = fudge_factor / args.ice_thickness

        fname = os.path.splitext(os.path.basename(fname))[0]

        # Set `num_particles` based on the user input (args.num_particles) with the following rules:
        # 1. If the user provides a value for `args.num_particles` and it is less than or equal to the `max_num_particles`, use it.
        # 2. If the user does not provide a value, use `rand_num_particles`.
        # 3. If the user's provided value exceeds `max_num_particles`, use `max_num_particles` instead.
        # 4. If the user specifies 'max', use max_num_particles, otherwise apply the existing conditions
        print_and_log(f"Trimming the mrc...", logging.INFO)
        rand_num_particles, max_num_particles = trim_vol_return_rand_particle_number(f"{structure_name}.mrc", f"{args.image_directory}/{fname}.mrc", args.scale_percent, f"{structure_name}.mrc")
        num_particles = (max_num_particles if str(args.num_particles).lower() == 'max' else
                 args.num_particles if args.num_particles and isinstance(args.num_particles, int) and args.num_particles <= max_num_particles else
                 rand_num_particles if not args.num_particles else
                 max_num_particles)


        if args.num_particles:
            print_and_log(f"Attempting to add {int(num_particles)} particles to the micrograph...", logging.INFO)
        else:
            print_and_log("Choosing a random number of particles to attempt to add to the micrograph...", logging.INFO)

        # Set the particle distribution type. If none is given (default), then 10% of the time it will choose random and 90% non-random
        dist_type = distribution if distribution else np.random.choice(['random', 'non-random'], p=[0.1, 0.9])
        if dist_type == 'non-random':
            # Randomly select a non-random distribution, weighted towards micrograph because it is the most realistic. Note: gaussian can create 1-5 gaussian blobs on the micrograph
            non_random_dist_type = np.random.choice(['circular', 'inverse circular', 'gaussian', 'micrograph'], p=[0.0025, 0.0075, 0.19, 0.8])
            if not args.num_particles:
                if non_random_dist_type == 'circular':
                    # Reduce the number of particles because the maximum was calculated based on the maximum number of particles that will fit in the micrograph side-by-side
                    num_particles = max(num_particles // 2, 2)  # Minimum number of particles is 2
                elif non_random_dist_type == 'inverse circular':
                    # Reduce the number of particles because the maximum was calculated based on the maximum number of particles that will fit in the micrograph side-by-side
                    num_particles = max(num_particles * 2 // 3, 2)
                elif non_random_dist_type == 'gaussian':
                    # Reduce the number of particles because the maximum was calculated based on the maximum number of particles that will fit in the micrograph side-by-side
                    num_particles = max(num_particles // 4, 2)
        else:
            non_random_dist_type = None

        print_and_log(f"Done! {num_particles} particles will be added to the micrograph.\n", logging.INFO)

        print_and_log(f"Projecting the structure volume {num_particles} times...", logging.INFO)
        # Determine orientation generator arguments based on user input
        if args.preferred_orientation:
            # Define the orientation generator based on user input and set fixed Euler angle
            orientgen_args = f"{args.orientgen_method}:n={num_particles}:phitoo={args.delta_angle}"
            # Add condition for fixed Euler angle (90 degrees could be interpreted as preferring the X-Y plane, for example)
            #if args.fixed_euler_angle == 90.0:
                # This example assumes you want to restrict altitude; adjust as necessary
                #orientgen_args += f":alt_min=89:alt_max=91"
        else:
            orientgen_args = f"rand:n={num_particles}:phitoo={args.phitoo}"

        # Modify the e2project3d.py command to use the new orientgen_args
        output = subprocess.run(["e2project3d.py", f"{structure_name}.mrc", f"--outfile=temp_{structure_name}.hdf", 
                        f"--orientgen={orientgen_args}", f"--parallel=thread:{args.cpus}"], capture_output=True, text=True).stdout

        #output = subprocess.run(["e2project3d.py", f"{structure_name}.mrc", f"--outfile=temp_{structure_name}.hdf", 
        #                f"--orientgen=rand:n={num_particles}:phitoo={args.phitoo}", f"--parallel=thread:{args.cpus}"], capture_output=True, text=True).stdout
        print_and_log(output, logging.INFO)
        print_and_log("Done!\n", logging.INFO)

        output = subprocess.run(["e2proc2d.py", f"temp_{structure_name}.hdf", f"temp_{structure_name}.mrc"], capture_output=True, text=True).stdout
        print_and_log(output, logging.INFO)
        print_and_log(f"Adding simulated noise to the particles by simulating {args.num_simulated_particle_frames} frames by sampling pixel values in each particle projection from a Poisson distribution...", logging.INFO)
        particles = readmrc(f"temp_{structure_name}.mrc")
        mean, gaussian_variance = estimate_noise_parameters(f"{args.image_directory}/{fname}.mrc")
        noisy_particles = add_poisson_noise(particles, args.num_simulated_particle_frames, args.cpus)
        writemrc(f"temp_{structure_name}_noise.mrc", noisy_particles)
        print_and_log("Done!\n", logging.INFO)

        print_and_log(f"Applying CTF based on the recorded defocus ({float(defocus):.4f} microns) and microscope parameters (Voltage: {args.voltage}keV, AmpCont: {args.ampcont}%, Cs: {args.Cs} mm, Pixelsize: {args.apix} Angstroms) that were used to collect the micrograph...", logging.INFO)
        output = subprocess.run(["e2proc2d.py", "--mult=-1", 
                        "--process", f"math.simulatectf:ampcont={args.ampcont}:bfactor=50:apix={args.apix}:cs={args.Cs}:defocus={defocus}:voltage={args.voltage}", 
                        "--process", "normalize.edgemean", f"temp_{structure_name}_noise.mrc", f"temp_{structure_name}_noise_CTF.mrc"], capture_output=True, text=True).stdout
        print_and_log(output, logging.INFO)
        print_and_log("Done!\n", logging.INFO)

        print_and_log(f"Adding the {num_particles} structure volume projections to the micrograph{f' {dist_type}ly' if dist_type else ''} while adding Gaussian (white) noise and simulating a relative ice thickness of {5/ice_thickness:.1f}...", logging.INFO)
        num_particles_projected, num_particles_with_saved_coordinates = add_images(f"{args.image_directory}/{fname}.mrc", f"temp_{structure_name}_noise_CTF.mrc", args.scale_percent, structure_name, args.border, args.edge_particles, args.save_edge_coordinates, ice_thickness, f"{structure_name}/{fname}_{structure_name}{repeat_suffix}", dist_type, non_random_dist_type, args.imod_coordinate_file, args.coord_coordinate_file, args.no_junk_filter, args.json_scale, args.flip_x, args.flip_y, args.polygon_expansion_distance, gaussian_variance, args.mrc, args.png, args.jpeg, args.jpeg_quality)
        print_and_log("Done!", logging.INFO)
        total_num_particles_projected += num_particles_projected
        total_num_particles_with_saved_coordinates += num_particles_with_saved_coordinates

        # Cleanup
        for temp_file in [f"temp_{structure_name}.hdf", f"temp_{structure_name}.mrc", f"temp_{structure_name}_noise.mrc", f"temp_{structure_name}_noise_CTF.mrc"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    # Downsample micrographs and coordinate files
    if args.binning > 1:
        # Downsample micrographs
        print_and_log(f"Binning/Downsampling micrographs by {args.binning} by Fourier cropping...\n", logging.INFO)
        parallel_downsample(f"{structure_name}/", args.cpus, args.binning)

        # Downsample coordinate files
        downsample_coordinate_files(structure_name, args.binning, args.imod_coordinate_file, args.coord_coordinate_file)

        if not args.keep:
            print_and_log("Removing non-downsamlpled micrographs...", logging.INFO)
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
    else:
        shutil.move(f"{structure_name}.star", f"{structure_name}/")

    # Log structure name, mass, number of micrographs generated, number of particles projected, and number of particles written to coordinate files
    with open("info.txt", "a") as f:
        # Check if the file is empty
        if f.tell() == 0:  # Only write the first line if the file is new
            f.write("structure_name mass(kDa) num_images num_particles_projected num_particles_saved\n")

        # Write the subsequent lines
        f.write(f"{structure_name} {mass} {args.num_images} {total_num_particles_projected} {total_num_particles_with_saved_coordinates}\n")
    
    box_size = get_mrc_box_size(f"{structure_name}.mrc")

    # Cleanup
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

    return total_num_particles_projected, total_num_particles_with_saved_coordinates, box_size

def main():
    start_time = time.time()

    # Parse user arguments, check them, and update them conditionally if necessary
    args = parse_arguments()

    # Loop over each provided structure and generate micrographs. Skip a structure if it doesn't download/exist
    total_structures = len(args.structures)
    total_number_of_particles_projected = 0
    total_number_of_particles_with_saved_coordinates = 0
    with ProcessPoolExecutor(max_workers=args.parallel_processes) as executor:
        # Prepare a list of tasks
        tasks = []
        for structure_index, structure_input in enumerate(args.structures):
            result = process_structure_input(structure_input, args.std_threshold, args.apix)
            if result:  # Check if result is not None
                structure_name, structure_type = result  # Now we're sure structure_name and structure_type are valid
                # Submit each task for execution
                task = executor.submit(generate_micrographs, args, structure_name, structure_type, structure_index, total_structures)
                tasks.append((task, structure_name))  # Store task with its associated structure_name
            else:
                print_and_log(f"Skipping structure due to an error or non-existence: {structure_input} (if you're trying to get a random structure, use the `-s r` flag)", logging.WARNING)

        # Wait for all tasks to complete, aggregate results, and crop particles if requested
        for task, structure_name in tasks:
            num_particles_projected, num_particles_with_saved_coordinates, box_size = task.result()
            total_number_of_particles_projected += num_particles_projected
            total_number_of_particles_with_saved_coordinates += num_particles_with_saved_coordinates

            # Check if cropping is enabled and perform cropping
            if args.crop_particles:
                box_size = args.box_size if args.box_size is not None else box_size
                crop_particles_from_micrographs(structure_name, box_size, args.cpus)

    end_time = time.time()
    time_str = time_diff(end_time - start_time)
    num_micrographs = args.num_images * len(args.structures)
    print_and_log("\n---------------------------------------------------------------------------------------------------------------------", logging.WARNING)
    print_and_log(f"Total generation time for {num_micrographs} micrograph{'s' if num_micrographs != 1 else ''} from {total_structures} structure{'s' if total_structures != 1 else ''} with the particle counts below: {time_str}", logging.WARNING)
    print_and_log(f"Total particles projected: {total_number_of_particles_projected}; Total particles saved to coordinate files: {total_number_of_particles_with_saved_coordinates}", logging.WARNING)
    print_and_log("---------------------------------------------------------------------------------------------------------------------\n", logging.WARNING)

    print_and_log("One .star file per structure can be found in the run directories.\n", logging.WARNING)

    if args.imod_coordinate_file:
        print_and_log("To open a micrograph with an IMOD coordinate file, run a command of this form:", logging.WARNING)
        print_and_log("3dmod image.mrc image.mod (Replace 'image.mrc' and 'image.mod' with your files)\n", logging.WARNING)

    if args.coord_coordinate_file:
        print_and_log("One (x y) .coord file per micrograph can be found in the run directories.\n", logging.WARNING)

    if args.crop_particles:
        print_and_log("Extracted particles can be found in the 'Particles' folder in the run directories.\n", logging.WARNING)

if __name__ == "__main__":
    main()
