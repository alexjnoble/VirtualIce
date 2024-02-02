#!/usr/bin/env python3
#
# Author: Alex J. Noble with help from GPT4, 2023-24 @SEMC, under the MIT License
#
# This script generates synthetic cryoEM micrographs given a list of noise micrographs and
# their corresponding defoci and PBD ids. It is intended that the noise micrographs are cryoEM
# images of buffer.
# Dependencies: EMAN2 installation (specifically e2pdb2mrc.py, e2project3d.py, and e2proc2d.py)
#               pip install mrcfile numpy scipy matplotlib cv2
#
# This program depends on EMAN2 to function properly. Users must separately
# install EMAN2 to use this program.
#
# EMAN2 is distributed under a dual license - BSD-3-Clause and GPL-2.0.
# For the details of EMAN2's licensing, please refer to:
# - BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
# - GPL-2.0: https://opensource.org/licenses/GPL-2.0
#
# You can obtain the EMAN2 source code from its official GitHub repository:
# https://github.com/cryoem/eman2
#
# Please ensure compliance with EMAN2's license terms when obtaining and using it.
__version__ = "1.0.0"

import os
import cv2
import glob
import json
import time
import random
import shutil
import mrcfile
import argparse
import textwrap
import itertools
import subprocess
import numpy as np
from datetime import timedelta
from matplotlib.path import Path
from multiprocessing import Pool
from urllib import request, error
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def check_num_particles(value):
    """
    Check if the number of particles is within the allowed range.
    This function exists just so that ./synthetic_micrograph_generator.py -h doesn't blow up.
    
    :param value: Number of particles.
    :return: Value if it is valid.
    :raises ArgumentTypeError: If the value is not in the allowed range.
    """
    ivalue = int(value)
    if ivalue < 2 or ivalue >= 1000000:
        raise argparse.ArgumentTypeError("Number of particles must be between 2 and 1000000")
    return ivalue

def check_binning(value):
    """
    Check if the binning is within the allowed range.
    This function exists just so that ./synthetic_micrograph_generator.py -h doesn't blow up.
    
    :param value: Binning requested.
    :return: Value if it is valid.
    :raises ArgumentTypeError: If the value is not in the allowed range.
    """
    ivalue = int(value)
    if ivalue < 2 or ivalue >= 100:
        raise argparse.ArgumentTypeError("Number of particles must be between 2 and 100")
    return ivalue

def print_verbose(message, verbosity, level=1):
    """
    Print messages depending on the verbosity level.

    :param str message: The message to be printed.
    :param int verbosity: The current verbosity level set by the user.
    :param int level: The level at which the message should be printed (default is 1).
    """
    if verbosity >= level:
        print(message)

def time_diff(time_diff):
    """
    Convert the time difference to a human-readable format.

    :param float time_diff: The time difference in seconds.
    :return: A formatted string indicating the time difference.
    """
    # Convert the time difference to a timedelta object
    delta = timedelta(seconds=time_diff)
    # Format the timedelta object based on its components
    if delta.days > 0:
        # If the time difference is more than a day, display days, hours, minutes, and seconds
        time_str = str(delta)
    else:
        # If the time difference is less than a day, display hours, minutes, and seconds
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{hours} hours, {minutes} minutes, {seconds} seconds"

    return time_str

def download_pdb(pdb_id, verbosity):
    """
    Download a PDB file from the RCSB website.

    :param str pdb_id: The ID of the PDB to be downloaded.
    :param int verbosity: The verbosity level for printing status messages.
    :return: True if the PDB exists, False if it doesn't.
    """
    print_verbose(f"Downloading PDB {pdb_id}...", verbosity)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        request.urlretrieve(url, f"{pdb_id}.pdb")
        print_verbose(f"Done!\n", verbosity)
        return True
    except error.HTTPError as e:
        print_verbose(f"Failed to download PDB {pdb_id}. HTTP Error: {e.code}\n", verbosity)
        return False
    except Exception as e:
        print_verbose(f"An unexpected error occurred while downloading PDB {pdb_id}. Error: {e}\n", verbosity)
        return False

def pdb_to_mrc(pdb_filename, mrc_filename, voxel_size, resolution, margin=5):
    """
    Convert a PDB file to an MRC file.
    
    :param pdb_filename: Path to the input PDB file.
    :param mrc_filename: Path to save the output MRC file.
    :param voxel_size: Desired voxel size of the MRC file in Angstroms.
    :param resolution: Desired resolution of the MRC file in Angstroms.
    :param margin: Additional margin to add around the protein in the grid.
    """
    # Define atomic weights and radii
    ATOM_DATA = {
        'H': {'radius': 1.20, 'mass': 1.008},
        'C': {'radius': 1.70, 'mass': 12.01},
        'N': {'radius': 1.55, 'mass': 14.01},
        'O': {'radius': 1.52, 'mass': 16.00},
        'S': {'radius': 1.80, 'mass': 32.07},
        'P': {'radius': 1.80, 'mass': 30.97},
        'Na': {'radius': 2.27, 'mass': 22.99},
        'Mg': {'radius': 1.73, 'mass': 24.31},
        'Cl': {'radius': 1.75, 'mass': 35.45},
        'K': {'radius': 2.75, 'mass': 39.10}, 
        'Ca': {'radius': 2.31, 'mass': 40.08},
        'Fe': {'radius': 2.00, 'mass': 55.85},
        'Zn': {'radius': 1.39, 'mass': 65.38},
        'Se': {'radius': 1.90, 'mass': 78.96},
        'Br': {'radius': 1.85, 'mass': 79.90},
        'I': {'radius': 1.98, 'mass': 126.90},
        'F': {'radius': 1.47, 'mass': 18.998},
        'Cu': {'radius': 1.4, 'mass': 63.546},
        'Mn': {'radius': 1.79, 'mass': 54.938},
        'Co': {'radius': 1.70, 'mass': 58.933},
        'Ni': {'radius': 1.62, 'mass': 58.693}
    }
    # Read all atoms from the pdb
    atoms = []
    with open(pdb_filename, 'r') as file:
        for line in file:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                element = line[76:78].strip()
                atoms.append((x, y, z, element))
    
    # Determine boundaries
    xs, ys, zs, _ = zip(*atoms)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    
    # Calculate grid size and make it cubic
    grid_size = np.ceil(np.array([max_x - min_x, max_y - min_y, max_z - min_z]) / voxel_size).astype(int) + 2 * margin
    max_size = np.max(grid_size)

    # Create an empty grid
    grid = np.zeros((max_size, max_size, max_size), dtype=np.float32)
    
    center = (np.array([min_x, min_y, min_z]) + np.array([max_x, max_y, max_z])) / 2.0
    
    # Populate the grid with atomic densities
    for x, y, z, element in atoms:
        atom_data = ATOM_DATA.get(element, ATOM_DATA['C'])  # Default to carbon if element not found
        radius = atom_data['radius']
        mass = atom_data['mass']
        
        # Convert coordinates to grid
        x, y, z = ((np.array([x, y, z]) - center) / voxel_size) + grid_size / 2.0
        
        # Populate the grid within the atom's radius with its mass
        for dx in range(int(-radius), int(radius) + 1):
            for dy in range(int(-radius), int(radius) + 1):
                for dz in range(int(-radius), int(radius) + 1):
                    if dx**2 + dy**2 + dz**2 <= radius**2:
                        grid[int(x)+dx, int(y)+dy, int(z)+dz] += mass
        
    # Apply Gaussian smoothing based on the resolution SIGMA IS BROKEN
    sigma = resolution / voxel_size
    #sigma = resolution / (2 * np.sqrt(2 * np.log(2)))
    grid = gaussian_filter(grid, sigma=sigma)
    
    # Save as MRC file
    with mrcfile.new(mrc_filename, overwrite=True) as mrc:
        mrc.set_data(grid)

def convert_pdb_to_mrc(pdb_id, apix, res, verbosity):
    """
    Convert a PDB file to MRC format using EMAN2's e2pdb2mrc.py script.

    :param str pdb_id: The ID of the PDB to be converted.
    :param float apix: The angstrom per pixel value to be used in the conversion.
    :param int res: The resolution to be used in the conversion.
    :param int verbosity: The verbosity level for printing status messages.

    :return: The mass extracted from the e2pdb2mrc.py script output.
    :rtype: int
    """
    print_verbose(f"Converting PDB {pdb_id} to MRC using EMAN2's e2pdb2mrc.py...", verbosity)
    cmd = ["e2pdb2mrc.py", "--apix", str(apix), "--res", str(res), "--center", f"{pdb_id}.pdb", f"{pdb_id}.mrc"]
    output = subprocess.run(cmd, capture_output=True, text=True)
    mass = int([line for line in output.stdout.split("\n") if "mass of" in line][0].split()[-2])
    print_verbose(f"Done!\n", verbosity)
    return mass

def readmrc(mrc_path):
    """
    Read an MRC file and return its data as a NumPy array.

    :param mrc_path: The file path of the MRC file to read.
    :return: The data of the MRC file as a NumPy array.
    """
    with mrcfile.open(mrc_path, mode='r') as mrc:
        data = mrc.data
        numpy_array = np.array(data)

    return numpy_array
    
def writemrc(mrc_path, numpy_array):
    """
    Write a NumPy array as an MRC file.

    :param mrc_path: The file path of the MRC file to write.
    :param numpy_array: The NumPy array to be written.
    """
    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(numpy_array)
    
    return

def write_star_header(file_basename):
    """
    Write the header for a .star file.

    :param str file_basename: The basename of the file to which the header should be written.
    """
    with open('%s.star' % file_basename, 'a') as the_file:
        the_file.write('\n')
        the_file.write('data_optics\n')
        the_file.write('\n')
        the_file.write('loop_\n')
        the_file.write('_rlnOpticsGroup #1\n')
        the_file.write('_rlnImageDimensionality #2\n')
        the_file.write('1 2\n')
        the_file.write('\n')
        the_file.write('data_particles\n')
        the_file.write('\n')
        the_file.write('loop_\n')
        the_file.write('_rlnMicrographName #1\n')
        the_file.write('_rlnCoordinateX #2\n')
        the_file.write('_rlnCoordinateY #3\n')
        the_file.write('_rlnPhaseShift #4\n')
        the_file.write('_rlnOpticsGroup #5\n')

def add_to_star(pdb_name, image_path, x_shift, y_shift):
    """
    Add entry to a STAR file with image path, x shift, and y shift.

    :param pdb_name: The name of the PDB file.
    :param image_path: The path of the image to add.
    :param x_shift: The x-shift value.
    :param y_shift: The y-shift value.
    """
    with open('%s.star' % pdb_name, 'a') as the_file:
        the_file.write('%s %s %s 0 1\n' % (image_path, x_shift, y_shift))
    
    return

def write_all_coordinates_to_star(pdb_name, image_path, particle_locations):
    """
    Write all particle locations to a STAR file.
    
    :param pdb_name: The name of the PDB file.
    :param image_path: The path of the image to add.
    :param particle_locations: A list of tuples, where each tuple contains the x, y coordinates.
    """
    # Open the star file once and write all coordinates
    with open(f'{pdb_name}.star', 'a') as the_file:
        for location in particle_locations:
            x_shift, y_shift = location
            the_file.write(f'{image_path} {x_shift} {y_shift} 0 1\n')

def convert_point_to_model(point_file, output_file):
    """
    Write an IMOD .mod file with particle coordinates.

    :param point_file: Path to the input .point file.
    :param output_file: Output file path for the .mod file.
    """
    try:
        # Run point2model command and give particles locations a circle of radius 3. Adjust the path if point2model is located elsewhere on your system.
        subprocess.run(["point2model", "-circle", "3", "-scat", point_file, output_file], check=True)
    except subprocess.CalledProcessError:
        print("Error while converting coordinates using point2model.")
    except FileNotFoundError:
        print("point2model not found. Ensure IMOD is installed and point2model is in your system's PATH.")

def write_mod_file(coordinates, output_file):
    """
    Write an IMOD .mod file with particle coordinates.

    :param coordinates: List of (x, y) coordinates for the particles.
    :param output_file: Output file path for the .mod file.
    """
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

    :param coordinates: List of (x, y) coordinates for the particles.
    :param output_file: Output file path for the .coord file.
    """
    coord_file = os.path.splitext(output_file)[0] + ".coord"
    with open(coord_file, 'w') as f:
        for x, y in coordinates:
            f.write(f"{x} {y}\n")  # Writing each coordinate as a new line in the .coord file

def read_polygons_from_json(json_file_path, expansion_distance, flip_x=False, flip_y=False, expand=True):
    """
    Read polygons from a JSON file generated by Anylabeling, optionally flip the coordinates,
    and optionally expand each polygon.
    
    :param json_file_path: Path to the JSON file.
    :param expansion_distance: Distance by which to expand the polygons.
    :param flip_x: Boolean to determine if the x-coordinates should be flipped.
    :param flip_y: Boolean to determine if the y-coordinates should be flipped.
    :param expand: Boolean to determine if the polygons should be expanded.
    :return: List of polygons where each polygon is a list of (x, y) coordinates.
    """
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

def shuf_images(num_images, image_list_file, verbosity):
    """
    Shuffle and select a specified number of random ice micrographs.

    :param int num_images: The number of images to select.
    :param str image_list_file: The path to the file containing the list of images.
    :param int verbosity: The verbosity level for printing status messages.
    :return: A list of selected ice micrograph filenames and defoci.
    """
    print_verbose(f"Selecting {num_images} random ice micrographs...", verbosity)
    # Read the contents of the image list file and store each image as a separate element in a list
    with open(image_list_file, "r") as f:
        image_list = f.readlines()
        image_list = [image.strip() for image in image_list]

    # Shuffle the order of images randomly
    random.shuffle(image_list)
    # Select the desired number of images from the shuffled list and sort alphanumerically
    selected_images = sorted(image_list[:num_images])

    print_verbose("Done!\n", verbosity)

    return selected_images

def non_uniform_random_number(min_val, max_val, threshold, weight):
    """
    Generate a non-uniform random number within a specified range.

    :param min_value: The minimum value of the range (inclusive).
    :param max_value: The maximum value of the range (inclusive).
    :param threshold: The threshold value for applying the weighted value.
    :param weighted_value: The weight assigned to values below the threshold.
    :return: A non-uniform random number within the specified range.
    """
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

    :param number: The starting number.
    :param primes: A list of prime numbers to consider.
    :param count: The number of primes to combine for finding the least common multiple.
    :return: The smallest number divisible by the combination of primes.
    """
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

    :return: Fourier cropped image.
    :rtype: numpy.ndarray
    :raises ValueError: If input image is not 2D or if downsample factor is not valid.
    """
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

    :return: None
    """
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
        print(f"Error processing {image_path}: {str(e)}")

def parallel_downsample(image_directory, cpus, downsample_factor):
    """
    Downsample all micrographs in a directory in parallel.

    :param str image_directory: Local directory name where the micrographs are stored in mrc, png, and/or jpeg formats.
    :param int downsample_factor: Factor by which to downsample the images in both dimensions.

    :return: None
    """
    image_extensions = ['.mrc', '.png', '.jpeg']
    image_paths = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if os.path.splitext(filename)[1].lower() in image_extensions]

    # Create a pool of worker processes
    pool = Pool(processes=cpus)

    # Apply the downsample_micrograph function to each image path in parallel
    pool.starmap(downsample_micrograph, zip(image_paths, itertools.repeat(downsample_factor)))

    # Close the pool to prevent any more tasks from being submitted
    pool.close()

    # Wait for all worker processes to finish
    pool.join()

def downsample_star_file(input_star, output_star, downsample_factor):
    """
    Read a STAR file, downsample the coordinates, and write a new STAR file.

    :param input_star: Path to the input STAR file.
    :param output_star: Path to the output STAR file.
    :param downsample_factor: Factor by which to downsample the coordinates.
    """
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

    :param input_point: Path to the input .point file.
    :param output_point: Path to the output .point file.
    :param downsample_factor: Factor by which to downsample the coordinates.
    """
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

    :param input_coord: Path to the input .coord file.
    :param output_coord: Path to the output .coord file.
    :param downsample_factor: Factor by which to downsample the coordinates.
    """
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

def trim_vol_return_rand_particle_number(input_mrc, input_micrograph, scale_percent, output_mrc):
    """
    Trim a volume and return a random number of particles within a micrograph based on the maximum
    number of projections of this volume that can fit in the micrograph without overlapping.

    :param input_mrc: The file path of the input volume in MRC format.
    :param input_micrograph: The file path of the input micrograph in MRC format.
    :param scale_percent: The percentage to scale the volume for trimming.
    :param output_mrc: The file path to save the trimmed volume.
    :return: A random number of particles up to a maximum of how many will fit in the micrograph.
    """
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
    
    # Choose a random number of particles between 1 and max, with low particle numbers (<100) downweighted
    rand_num_particles = non_uniform_random_number(2, max_num_particles, 100, 0.5)
    
    return rand_num_particles, max_num_particles

def filter_coordinates_outside_polygons(particle_locations, json_scale, polygons):
    """
    Filters out particle locations that are inside any polygon.
    
    :param particle_locations: List of (x, y) coordinates of particle locations.
    :param json_scale: Binning factor used when labeling junk to create the json file.
    :param polygons: List of polygons where each polygon is a list of (x, y) coordinates.
    :return: List of (x, y) coordinates of particle locations that are outside the polygons.
    """
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

def generate_particle_locations(image_size, num_small_images, half_small_image_width, border_distance, dist_type, non_random_dist_type):
    """
    Generate random locations for particles within an image.

    :param image_size: The size of the image as a tuple (width, height).
    :param num_small_images: The number of small images or particles to generate coordinates for.
    :param half_small_image_width: Half the width of a small image.
    :param border_distance: The minimum distance between particles and the image border.
    :param dist_type: Particle location generation distribution type - 'random' or 'non-random'.
    :param non_random_dist_type: Type of non-random distribution when dist_type is 'non-random' - 'circular' or 'gaussian'.
    :return: A list of particle locations as tuples (x, y).
    """
    width, height = image_size
    particle_locations = []

    def is_far_enough(new_particle_location, particle_locations, half_small_image_width):
        """
        Check if a new particle location is far enough from existing particle locations.

        :param new_particle_location: The new particle location as a tuple (x, y).
        :param particle_locations: The existing particle locations.
        :param half_small_image_width: Half the width of a small image.
        :return: True if the new particle location is far enough, False otherwise.
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
            # Make a curcular cluster of particles
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
        
    return particle_locations

def process_slice(args):
    """
    Process a slice of the particle stack by adding Poisson and Gaussian electronic noise.
    
    :param args: A tuple containing the following parameters:
                 - slice: A 3D numpy array representing a slice of the particle stack.
                 - num_frames: Number of frames to simulate for each particle image.
                 - scaling_factor: Factor by which to scale the particle images before adding noise.
                 - electronic_noise_std: Standard deviation of the Gaussian electronic noise.
    :return: A 3D numpy array representing the processed slice of the particle stack with added noise.
    """
    
    # Unpack the arguments
    slice, num_frames, scaling_factor, electronic_noise_std = args
    
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
            
            # Add Poisson noise to the non-zero values in the particle
            noisy_frame[mask] = np.random.poisson(particle[mask])
            
            # Generate Gaussian electronic noise and restrict it to the mask
            electronic_noise = np.round(np.random.normal(loc=0, scale=electronic_noise_std, size=particle.shape)).astype(np.int32)
            electronic_noise *= mask.astype(np.int32)
            
            # Add the electronic noise to the noisy frame
            noisy_frame += electronic_noise
            
            # Accumulate the noisy frame to the noisy slice
            noisy_slice[i, :, :] += noisy_frame
            
    return noisy_slice

def add_combined_noise(particle_stack, num_frames, num_cores, scaling_factor=1.0, electronic_noise_std=1.0):
    """
    Add Poisson and Gaussian electronic noise to a stack of particle images.
    
    This function simulates the acquisition of `num_frames` frames for each particle image
    in the input stack, adds Poisson noise and Gaussian electronic noise to each frame,
    and then sums up the frames to obtain the final noisy particle image. The function
    applies both noises only to the non-zero values in each particle image, preserving
    the background.
    
    :param particle_stack: 3D numpy array representing a stack of 2D particle images.
    :param num_frames: Number of frames to simulate for each particle image.
    :param num_cores: Number of CPU cores to parallelize slices across.
    :param scaling_factor: Factor by which to scale the particle images before adding noise.
    :param electronic_noise_std: Standard deviation of the Gaussian electronic noise.
    :return: 3D numpy array representing the stack of noisy particle images.
    """
    # Split the particle stack into slices
    slices = np.array_split(particle_stack, num_cores)
    
    # Prepare the arguments for each slice
    args = [(s, num_frames, scaling_factor, electronic_noise_std) for s in slices]
    
    # Create a pool of worker processes
    with Pool(num_cores) as pool:
        # Process each slice in parallel
        noisy_slices = pool.map(process_slice, args)
    
    # Concatenate the processed slices back into a single stack
    noisy_particle_stack = np.concatenate(noisy_slices, axis=0)
    
    return noisy_particle_stack

def create_collage(large_image, small_images, particle_locations):
    """
    Create a collage of small images on a blank canvas of the same size as the large image.
    
    :param large_image_shape: Shape of the large image.
    :param small_images: List of small images to place on the canvas.
    :param particle_locations: Coordinates where each small image should be placed.
    :return: Collage of small images.
    """
    collage = np.zeros(large_image.shape, dtype=large_image.dtype)
    
    for i, small_image in enumerate(small_images):
        x, y = particle_locations[i]
        x_start = x - small_image.shape[0] // 2
        y_start = y - small_image.shape[1] // 2
        
        x_end = x_start + small_image.shape[0]
        y_end = y_start + small_image.shape[1]
        
        collage[y_start:y_end, x_start:x_end] += small_image
    
    return collage

def blend_images(large_image, small_images, particle_locations, scale, pdb_name, imod_coordinate_file, coord_coordinate_file, large_image_path, image_path, json_scale, flip_x, flip_y, polygon_expansion_distance):
    """
    Blend small images (particles) into a large image (micrograph).
    Also makes coordinate files.

    :param large_image: The large image where small images will be blended into.
    :param small_images: The list of small images or particles to be blended.
    :param particle_locations: The locations of the particles within the large image.
    :param scale: The scale factor to adjust the intensity of the particles.
    :param pdb_name: The name of the PDB file.
    :param image_path: The path of the image.
    :param json_scale: Binning factor used when labeling junk to create the json file.
    :param flip_x: Boolean to determine if the x-coordinates should be flipped.
    :param flip_y: Boolean to determine if the y-coordinates should be flipped.
    :param polygon_expansion_distance: Distance by which to expand the polygons.
    :return: The blended large image.
    """
    json_file_path = os.path.splitext(large_image_path)[0] + ".json"
    if os.path.exists(json_file_path):
        polygons = read_polygons_from_json(json_file_path, polygon_expansion_distance, flip_x, flip_y, expand=True)
        
        # Remove particle locations from inside polygons (junk in micrographs) when writing coordinate files
        filtered_particle_locations = filter_coordinates_outside_polygons(particle_locations, json_scale, polygons)
        num_particles_removed = len(particle_locations) - len(filtered_particle_locations)
        removed_particles = [item for item in particle_locations if item not in filtered_particle_locations]
        print(f"{num_particles_removed} particles removed from coordinate file(s) based on the corresponding JSON file.")
    else:
        print(f"JSON file with polygons for bad micrograph areas not found: {json_file_path}")
        filtered_particle_locations = particle_locations

    # Normalize the input micrograph to itself
    large_image[:, :] = (large_image[:, :] - large_image[:, :].mean())/large_image[:, :].std()
    
    collage = create_collage(large_image, small_images, particle_locations)
    collage *= scale  # Apply scaling if necessary
    
    blended_image = large_image + collage  # Blend the collage with the large image

    # Normalize the resulting micrograph to itself
    blended_image = (blended_image - blended_image.mean()) / blended_image.std()

    write_all_coordinates_to_star(pdb_name, image_path, filtered_particle_locations)
    
    # Make an Imod .mod coordinates file if requested
    if imod_coordinate_file:
        write_mod_file(filtered_particle_locations, os.path.splitext(image_path)[0] + ".mod")
        write_mod_file(removed_particles, os.path.splitext(image_path)[0] + "r.mod")

    # Make a .coord coordinates file if requested
    if coord_coordinate_file:
        write_coord_file(filtered_particle_locations, os.path.splitext(image_path)[0] + ".coord")

    return blended_image

def blend_images2(large_image, small_images, particle_locations, scale, pdb_name, imod_coordinate_file, coord_coordinate_file, large_image_path, image_path, flip_x, flip_y):
    """
    Blend small images (particles) into a large image (micrograph).
    Also makes coordinate files.

    :param large_image: The large image where small images will be blended into.
    :param small_images: The list of small images or particles to be blended.
    :param particle_locations: The locations of the particles within the large image.
    :param scale: The scale factor to adjust the intensity of the particles.
    :param pdb_name: The name of the PDB file.
    :param image_path: The path of the image.
    :param flip_x: Boolean to determine if the x-coordinates should be flipped.
    :param flip_y: Boolean to determine if the y-coordinates should be flipped.
    :return: The blended large image.
    """
    json_file_path = os.path.splitext(large_image_path)[0] + ".json"
    if os.path.exists(json_file_path):
        polygons = read_polygons_from_json(json_file_path, flip_x, flip_y)
        
        # Remove particle locations from inside polygons (junk in micrographs) when writing coordinate files
        filtered_particle_locations = filter_coordinates_outside_polygons(particle_locations, polygons)
    else:
        print(f"JSON file with polygons for bad micrograph areas not found: {json_file_path}")
        filtered_particle_locations = particle_locations

    # Normalize the input micrograph to itself
    large_image[:, :] = (large_image[:, :] - large_image[:, :].mean())/large_image[:, :].std()
    
    # Add each particle to the micrograph
    for i in range(len(small_images)):
        particle = small_images[i]
        
        # Normalize the particle stack to itself
        particle[:, :] = (particle[:, :] - particle[:, :].mean())/particle[:, :].std()
    
        # Shift offset by half of the particle box size
        x_offset = particle_locations[i][0]
        y_offset = particle_locations[i][1]
        x_shift = x_offset - int(particle.shape[0]/2)
        y_shift = y_offset - int(particle.shape[0]/2)
        
        # Fudge factor scale to make the protein densities realistically visible
        particle = scale * particle

        # Blend the particle with the micrograph
        large_image[y_shift: y_shift + particle.shape[0], x_shift: x_shift + particle.shape[1]] = \
            large_image[y_shift: y_shift + particle.shape[0], x_shift: x_shift + particle.shape[1]] + \
            particle[:, :]
        
        # Add coordinates to star file
        add_to_star(pdb_name, image_path, x_offset, y_offset)
    
    # Make an Imod .mod coordinates file if requested
    if imod_coordinate_file:
        write_mod_file(filtered_particle_locations, os.path.splitext(image_path)[0] + ".mod")

    # Make a .coord coordinates file if requested
    if coord_coordinate_file:
        write_coord_file(filtered_particle_locations, os.path.splitext(image_path)[0] + ".coord")

    # Normalize the resulting micrograph to itself
    large_image[:, :] = (large_image[:, :] - large_image[:, :].mean())/large_image[:, :].std()
    
    return large_image

def add_images(large_image_path, small_images, pdb_id, border_distance, scale, output_path, dist_type, non_random_dist_type, imod_coordinate_file, coord_coordinate_file, json_scale, flip_x, flip_y, polygon_expansion_distance, save_as_mrc, save_as_png, save_as_jpeg, jpeg_quality, verbosity):
    """
    Add small images or particles to a large image and save the resulting micrograph.

    :param large_image_path: The file path of the large image or micrograph.
    :param small_images: The file path of the small images or particles.
    :param pdb_id: The ID of the PDB file.
    :param border_distance: The minimum distance between particles and the image border.
    :param output_path: The file path to save the resulting micrograph.
    :param scale: The scale factor to adjust the intensity of the particles.
    :param flip_x: Boolean to determine if the x-coordinates should be flipped.
    :param flip_y: Boolean to determine if the y-coordinates should be flipped.
    :param num_cores: Number of processes to use for multiprocessing.
    :param verbosity: Print out verbosity level.
    :return: The actual number of particles added to the micrograph.
    """
    # Read micrograph and particles, and get some information
    large_image = readmrc(large_image_path)
    small_images = readmrc(small_images)
    image_size = np.flip(large_image.shape)
    num_small_images = len(small_images)

    # Don't let particles be closer to the edge of the micrograph than half of the particle sidelength
    half_small_image_width = int(small_images.shape[1]/2)
    border_distance = max(border_distance, small_images.shape[1]/2)
        
    particle_locations = generate_particle_locations(image_size, num_small_images, half_small_image_width, border_distance, dist_type, non_random_dist_type)
    
    # Blend the images together
    if len(particle_locations) == num_small_images:
        result_image = blend_images(large_image, small_images, particle_locations, scale, pdb_id, imod_coordinate_file, coord_coordinate_file, large_image_path, output_path, json_scale, flip_x, flip_y, polygon_expansion_distance)
    else:
        print_verbose(f"Only {len(particle_locations)} could fit into the image. Adding those to the micrograph now...", verbosity)
        result_image = blend_images(large_image, small_images[:len(particle_locations), :, :], particle_locations, scale, pdb_id, imod_coordinate_file, coord_coordinate_file, large_image_path, output_path, json_scale, flip_x, flip_y, polygon_expansion_distance)

    # Save the resulting micrograph in specified formats
    if save_as_mrc:
        print_verbose(f"\nWriting synthetic micrograph as a MRC file: {output_path}.mrc...\n", verbosity)
        writemrc(output_path + '.mrc', (result_image - np.mean(result_image)) / np.std(result_image))  # Write mrc normalized with mean of 0 and std of 1
    if save_as_png:
        # Needs to be scaled from 0 to 255 and flipped
        result_image -= result_image.min()
        result_image = result_image / result_image.max() * 255.0
        print_verbose(f"\nWriting synthetic micrograph as a PNG file: {output_path}.png...\n", verbosity)
        cv2.imwrite(output_path + '.png', np.flip(result_image, axis=0))
    if save_as_jpeg:
        # Needs to be scaled from 0 to 255 and flipped
        result_image -= result_image.min()
        result_image = result_image / result_image.max() * 255.0
        print_verbose(f"\nWriting synthetic micrograph as a JPEG file: {output_path}.jpeg...\n", verbosity)
        cv2.imwrite(output_path + '.jpeg', np.flip(result_image, axis=0), [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    return len(particle_locations)

def generate_micrographs(pdb_id, args):
    """
    Generate synthetic micrographs for a specified PDB ID.

    This function orchestrates the workflow for generating synthetic micrographs
    for a given PDB ID. It performs PDB download, conversion, image shuffling,
    and iterates through the generation process for each selected image. It also
    handles cleanup operations post-generation.

    :param pdb_id: str
        The PDB ID for which synthetic micrographs are to be generated. This should
        be a valid PDB identifier corresponding to the desired protein structure.
    :param args: Namespace
        The argument namespace containing all the command-line arguments specified by the user.
    
    :return: The total number of particles actually added to all of the micrographs for the given pdb.
    """
    # Convert short distribution to full distribution name
    distribution_mapping = {'r': 'random', 'n': 'non-random'}
    distribution = distribution_mapping.get(args.distribution, args.distribution)

    # Create output directory
    if not os.path.exists(pdb_id):
        os.mkdir(pdb_id)

    # Convert PDB to MRC
    mass = convert_pdb_to_mrc(pdb_id, args.apix, args.pdb_to_mrc_resolution, args.verbosity)
    print_verbose(f"Calculated mass for PDB {pdb_id}: {mass} kDa", args.verbosity)
    
    # Write STAR header for the current synthetic dataset
    write_star_header(pdb_id)
    
    # Shuffle ice images
    selected_images = shuf_images(args.num_images, args.image_list_file, args.verbosity)
    
    # Main Loop
    total_num_particles = 0
    for line in selected_images:
        # Parse the 'micrograph_name.mrc defocus' line
        fields = line.strip().split()
        fname, defocus = fields[0], fields[1]
        
        extra_hyphens = '-' * len(fname)
        print_verbose(f"\n----------------------------------------{extra_hyphens}", args.verbosity)
        print_verbose(f"Generating synthetic micrograph from {fname}...", args.verbosity)
        print_verbose(f"----------------------------------------{extra_hyphens}\n", args.verbosity)
        
        # Random ice thickness
        # Adjust the relative ice thickness to work mathematically (yes, the inputs are completely inversely named just so the user gets a number that feels right)
        min_ice_thickness = 5/args.max_ice_thickness
        max_ice_thickness = 5/args.min_ice_thickness
        rand_ice_thickness = random.uniform(min_ice_thickness, max_ice_thickness)

        fname = os.path.splitext(os.path.basename(fname))[0]
        
        # Set `num_particles` based on the user input (args.num_particles) with the following rules:
        # 1. If the user provides a value for `args.num_particles` and it is less than or equal to the `max_num_particles`, use it.
        # 2. If the user does not provide a value, use `rand_num_particles`.
        # 3. If the user's provided value exceeds `max_num_particles`, use `max_num_particles` instead.
        rand_num_particles, max_num_particles = trim_vol_return_rand_particle_number(f"{pdb_id}.mrc", f"{args.image_directory}/{fname}.mrc", args.scale_percent, f"{pdb_id}.mrc")
        num_particles = args.num_particles if args.num_particles and args.num_particles <= max_num_particles else (rand_num_particles if not args.num_particles else max_num_particles)

        if args.num_particles:
            print_verbose(f"Trimming the generated mrc volume from the pdb and adding {int(num_particles)} particles to the micrograph...", args.verbosity)
        else:
            print_verbose("Trimming the generated mrc volume from the pdb and choosing a random number of particles to add to the micrograph...", args.verbosity)
        
        # Set the particle distribution type. If none is given (default), then 30% of the time it will choose random and 70% non-random
        dist_type = distribution if distribution else np.random.choice(['random', 'non-random'], p=[0.3, 0.7])
        if dist_type == 'non-random':
            # Randomly select a non-random distribution, weighted 7:1 to gaussian distributions because it can create 1-6 gaussian blobs on the micrograph
            non_random_dist_type = np.random.choice(['circular', 'gaussian'], p=[0.14, 0.86])
            if non_random_dist_type == 'circular':
                # Reduce the number of particles because the maximum was calculated based on the maximum number of particles that will fit in the micrograph side-by-side
                num_particles = max(num_particles // 2, 2)
            elif non_random_dist_type == 'gaussian':
                # Reduce the number of particles because the maximum was calculated based on the maximum number of particles that will fit in the micrograph side-by-side
                num_particles = max(num_particles // 4, 2)
        else:
            non_random_dist_type = None
        
        print_verbose(f"Done! {num_particles} particles will be added to the micrograph.\n", args.verbosity)
        
        print_verbose(f"Projecting the PDB volume {num_particles} times...", args.verbosity)
        output = subprocess.run(["e2project3d.py", f"{pdb_id}.mrc", f"--outfile=temp_{pdb_id}.hdf", 
                        f"--orientgen=rand:n={num_particles}:phitoo={args.phitoo}", f"--parallel=thread:{args.cpus}"], capture_output=True, text=True).stdout
        print_verbose(output, args.verbosity)
        print_verbose("Done!\n", args.verbosity)

        subprocess.run(["e2proc2d.py", f"temp_{pdb_id}.hdf", f"temp_{pdb_id}.mrc"], capture_output=True, text=True).stdout
        print_verbose(f"Adding simulated noise to the particles by simulating {args.num_simulated_particle_frames} frames by sampling pixel values in each particle from a Poisson distribution and adding Gaussian (white) noise...", args.verbosity)
        particles = readmrc(f"temp_{pdb_id}.mrc")
        noisy_particles = add_combined_noise(particles, args.num_simulated_particle_frames, args.cpus, 0.3)
        writemrc(f"temp_{pdb_id}_noise.mrc", noisy_particles)
        print_verbose("Done!\n", args.verbosity)
        
        print_verbose(f"Applying CTF based on the recorded defocus ({float(defocus):.4f} microns) and microscope parameters (Voltage: {args.voltage}keV, AmpCont: {args.ampcont}%, Cs: {args.Cs} mm, Pixelsize: {args.apix} Angstroms) that were used to collect the micrograph...", args.verbosity)
        output = subprocess.run(["e2proc2d.py", "--mult=-1", 
                        "--process", f"math.simulatectf:ampcont={args.ampcont}:bfactor=50:apix={args.apix}:cs={args.Cs}:defocus={defocus}:voltage={args.voltage}", 
                        "--process", "normalize.edgemean", f"temp_{pdb_id}_noise.mrc", f"temp_{pdb_id}_noise_CTF.mrc"], capture_output=True, text=True).stdout
        print_verbose(output, args.verbosity)
        print_verbose("Done!\n", args.verbosity)
        
        print_verbose(f"Adding the {num_particles} PBD volume projections to the micrograph{f' {dist_type}ly' if dist_type else ''} while simulating a relative ice thickness of {5/rand_ice_thickness:.1f}...", args.verbosity)
        num_particles = add_images(f"{args.image_directory}/{fname}.mrc", f"temp_{pdb_id}_noise_CTF.mrc", pdb_id, args.border, rand_ice_thickness, f"{pdb_id}/{fname}_{pdb_id}", dist_type, non_random_dist_type, args.imod_coordinate_file, args.coord_coordinate_file, args.json_scale, args.flip_x, args.flip_y, args.polygon_expansion_distance, args.mrc, args.png, args.jpeg, args.jpeg_quality, args.verbosity)
        print_verbose("Done!", args.verbosity)
        total_num_particles += num_particles

        # Cleanup
        for temp_file in [f"temp_{pdb_id}.hdf", f"temp_{pdb_id}.mrc", f"temp_{pdb_id}_noise.mrc", f"temp_{pdb_id}_noise_CTF.mrc"]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # Downsample micrographs and coordinate files
    if args.binning > 1:
        # Downsample micrographs
        print_verbose(f"Binning/Downsampling micrographs by {args.binning} by Fourier cropping...\n", args.verbosity)
        parallel_downsample(f"{pdb_id}/", args.cpus, args.binning)
        
        # Downsample coordinate files
        downsample_star_file(f"{pdb_id}.star", f"{pdb_id}_bin{args.binning}.star", args.binning)
        if args.imod_coordinate_file:
            for filename in os.listdir(f"{pdb_id}/"):
                if filename.endswith(".point"):
                    input_file = os.path.join(f"{pdb_id}/", filename)
                    output_point_file = os.path.join(f"{pdb_id}/bin_{args.binning}/", filename.replace('.point', f'_bin{args.binning}.point'))
                    # First downsample the .point files
                    downsample_point_file(input_file, output_point_file, args.binning)
                    # Then convert all of the .point files to .mod files
                    mod_file = os.path.splitext(output_point_file)[0] + ".mod"
                    convert_point_to_model(output_point_file, mod_file)
        if args.coord_coordinate_file:
            for filename in os.listdir(f"{pdb_id}/"):
                if filename.endswith(".coord"):
                    input_file = os.path.join(f"{pdb_id}/", filename)
                    output_coord_file = os.path.join(f"{pdb_id}/bin_{args.binning}/", filename.replace('.coord', f'_bin{args.binning}.coord'))
                    # First downsample the .coord files
                    downsample_coord_file(input_file, output_coord_file, args.binning)

        if not args.keep:
            # Delete the non-downsampled micrographs and coordinate files and move the binned ones to the parent directory
            print_verbose("Removing non-downsamlpled micrographs...", args.verbosity)
            bin_dir = f"{pdb_id}/bin_{args.binning}/"
            for file in glob.glob(f"{pdb_id}/*.mrc"):
                os.remove(file)
            for file in glob.glob(f"{pdb_id}/*.png"):
                os.remove(file)
            for file in glob.glob(f"{pdb_id}/*.jpeg"):
                os.remove(file)
            for file in glob.glob(f"{pdb_id}/*.mod"):
                os.remove(file)
            for file in glob.glob(f"{pdb_id}/*.coord"):
                os.remove(file)
            for file in glob.glob(f"{pdb_id}/bin_{args.binning}/*.mrc"):
                shutil.move(file, f"{pdb_id}/")
            for file in glob.glob(f"{pdb_id}/bin_{args.binning}/*.png"):
                shutil.move(file, f"{pdb_id}/")
            for file in glob.glob(f"{pdb_id}/bin_{args.binning}/*.jpeg"):
                shutil.move(file, f"{pdb_id}/")
            for file in glob.glob(f"{pdb_id}/bin_{args.binning}/*.mod"):
                shutil.move(file, f"{pdb_id}/")
            for file in glob.glob(f"{pdb_id}/bin_{args.binning}/*.coord"):
                shutil.move(file, f"{pdb_id}/")
            shutil.rmtree(f"{pdb_id}/bin_{args.binning}/")
            shutil.move(f"{pdb_id}_bin{args.binning}.star", f"{pdb_id}/")
            os.remove(f"{pdb_id}.star")
        else:
            shutil.move(f"{pdb_id}.star", f"{pdb_id}/")
            shutil.move(f"{pdb_id}_bin{args.binning}.star", f"{pdb_id}/bin_{args.binning}/")
    else:
        shutil.move(f"{pdb_id}.star", f"{pdb_id}/")

    # Log PDB id, mass, and number of micrographs generated
    with open("pdb_mass_numimages_numparticles.txt", "a") as f:
        f.write(f"{pdb_id} {mass} {args.num_images} {total_num_particles}\n")

    # Cleanup
    for directory in [f"{pdb_id}/", f"{pdb_id}/bin_{args.binning}/"]:
        try:
            for file_name in os.listdir(directory):
                if file_name.endswith(".point"):
                    file_path = os.path.join(directory, file_name)
                    os.remove(file_path)
        except FileNotFoundError:
            pass
    for temp_file in [f"{pdb_id}.pdb", f"{pdb_id}.mrc", "thread.out", ".eman2log.txt"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return total_num_particles
    
def main():
    start_time = time.time()
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Feature-rich synthetic cryoEM micrograph generator that projects pdbs>mrcs onto existing buffer cryoEM micrographs. Star files for particle coordinates are outputed by default, mod and coord files are optional. Particle coordinates located within per-micrograph polygons are projected but not written to coordinate files.")
    parser.add_argument("-p", "--pdbs", type=str, nargs='+', default=['1TIM'], help="PDB ID(s) for the protein structure(s). Default is 1TIM.")
    parser.add_argument("-n", "--num_images", type=int, default=100, help="Number of micrographs to create")
    parser.add_argument("-N", "--num_particles", type=check_num_particles, help="Number of particles to project onto the micrograph after random rotation. Default is a random number (weighted to favor numbers above 100 twice as much as below 100) up to a maximum of the number of particles that can fit into the micrograph without overlapping.")
    parser.add_argument("-d", "--distribution", type=str, choices=['r', 'random', 'n', 'non-random'], default=None, help="Distribution type for generating particle locations: 'random' (or 'r') and 'non-random' (or 'n'). Default is None which randomly selects a distribution per micrograph.")
    parser.add_argument("-m", "--min_ice_thickness", type=float, default=30, help="Minimum ice thickness, which scales how much the particle is added to the image (this is a relative value)")
    parser.add_argument("-M", "--max_ice_thickness", type=float, default=90, help="Maximum ice thickness, which scales how much the particle is added to the image (this is a relative value)")
    parser.add_argument("-c", "--cpus", type=int, default=os.cpu_count(), help="Number of CPUs to use for various processing steps. Default is the number of CPU cores available")
    parser.add_argument("-i", "--image_list_file", type=str, default="ice_images/good_images_with_defocus.txt", help="File containing filenames of images with a defocus value after each filename (space between)")
    parser.add_argument("-D", "--image_directory", type=str, default="ice_images", help="Local directory name where the micrographs are stored in mrc format")
    parser.add_argument("--mrc", action="store_true", default=True, help="Save micrographs as .mrc (default if no format is specified)")
    parser.add_argument("--no-mrc", dest="mrc", action="store_false", help="Do not save micrographs as .mrc")
    parser.add_argument("-P", "--png", action="store_true", help="Save micrographs as .png")
    parser.add_argument("-J", "--jpeg", action="store_true", help="Save micrographs as .jpeg")
    parser.add_argument("-Q", "--jpeg-quality", type=int, default=95, help="Quality of saved .jpeg images (0 to 100)")
    parser.add_argument("-j", "--json_scale", type=int, default=4, help="Binning factor used when labeling junk to create the json file.")
    parser.add_argument("-x", "--flip_x", action="store_true", help="Flip the polygons that identify junk along the x-axis")
    parser.add_argument("-y", "--flip_y", action="store_true", help="Flip the polygons that identify junk along the y-axis")
    parser.add_argument("-e", "--polygon_expansion_distance", type=int, default=5, help="Number of pixels to expand each polygon in the json file that defines areas to not place particle coordinates. The size of the pixels used here is the same size as the pixels that the json file uses.")
    parser.add_argument("-b", "--binning", type=check_binning, default=1, help="Bin/Downsample the micrographs by Fourier cropping after superimposing particle projections. Binning is the sidelength divided by this factor (e.g. -d 4 for a 4k x 4k micrograph will result in a 1k x 1k micrograph)")
    parser.add_argument("-k", "--keep", action="store_true", help="Keep the non-downsampled micrographs if downsampling is requested. Non-downsampled micrographs are deleted by default")
    parser.add_argument("-a", "--apix", type=float, default=1.096, help="Pixel size of the existing images (EMAN2 e2pdb2mrc.py option)")
    parser.add_argument("-r", "--pdb_to_mrc_resolution", type=float, default=3, help="Resolution in Angstroms for PDB to MRC conversion (EMAN2 e2pdb2mrc.py option)")
    parser.add_argument("-f", "--num_simulated_particle_frames", type=int, default=50, help="Number of simulated particle frames to generate Poisson noise")
    parser.add_argument("-F", "--phitoo", type=float, default=0.1, help="Phitoo value for random 3D projection (EMAN2 e2project3d.py option). This is the angular step size for rotating before projecting")
    parser.add_argument("-A", "--ampcont", type=float, default=10, help="Amplitude contrast when applying CTF to projections (EMAN2 e2proc2d.py option)")
    parser.add_argument("--Cs", type=float, default=0.001, help="Microscope spherical aberration when applying CTF to projections (EMAN2 e2proc2d.py option). Default is 0.001 because the microscope used to collect the provided buffer cryoEM micrographs has a Cs corrector")
    parser.add_argument("-K", "--voltage", type=float, default=300, help="Microscope voltage when applying CTF to projections (EMAN2 e2proc2d.py option)")
    parser.add_argument("-s", "--scale_percent", type=float, default=33.33, help="How much larger to make the resulting mrc file from the pdb file compared to the minimum equilateral cube (default: 33.33; ie. 33.33% larger)")
    parser.add_argument("-B", "--border", type=int, default=0, help="Minimum distance of center of particles from the image border (default: 0 = reverts to half boxsize)")
    parser.add_argument("-I", "--imod_coordinate_file", action="store_true", help="Also output one IMOD .mod coordinate file per micrograph. Note: IMOD must be installed and working")
    parser.add_argument("-C", "--coord_coordinate_file", action="store_true", help="Also output one .coord coordinate file per micrograph")
    parser.add_argument("-V", "--verbosity", type=int, default=2, help="Set verbosity level: 0 (quiet), 1 (some output), 2 (verbose). Default is 2 (verbose).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Set verbosity to 0 (quiet). Overrides --verbosity if both are provided.")
    parser.add_argument("-v", "--version", action="version", version=f"Synthetic Micrograph Generator v{__version__}")
    args = parser.parse_args()
    
    if not (args.mrc or args.png or args.jpeg):
        parser.error("No format specified for saving images. Please specify at least one format.")

    # Set verbosity level
    args.verbosity = 0 if args.quiet else args.verbosity

    # Limit num_images to: min = num of files in image_list_file, max = number of .mrc files in the image_directory
    args.num_images = min(args.num_images, sum(1 for _ in open(args.image_list_file)))
    args.num_images = min(args.num_images, sum(1 for file in os.listdir(args.image_directory) if file.endswith('.mrc')))

    # Print all arguments for the user's information
    formatted_output = ""
    for arg, value in vars(args).items():
        formatted_output += f"{arg}: {value};\n"

    # Wrap the output text to fit in rows and columns
    argument_printout = textwrap.fill(formatted_output, width=80)

    print("-----------------------------------------------------------------------------------------------")
    print(f"Generating {args.num_images} synthetic micrographs for each PDB ({args.pdbs}) using micrographs in {args.image_directory}/ ...\n")
    print("Synthetic Micrograph Generator arguments:\n")
    print(argument_printout)
    print("-----------------------------------------------------------------------------------------------\n")

    # Loop over each provided PDB ID. Skip a PDB if it doesn't download
    total_number_of_particles = 0
    for pdb_id in args.pdbs:
        if download_pdb(pdb_id, args.verbosity):
            number_of_particles = generate_micrographs(pdb_id, args)
            total_number_of_particles += number_of_particles
        else:
            print_verbose(f"Skipping {pdb_id} due to download failure.\n", args.verbosity)
    
    end_time = time.time()
    time_str = time_diff(end_time - start_time)
    num_micrographs = args.num_images * len(args.pdbs)
    print(f"\nTotal time taken to generate {num_micrographs} synthetic micrograph{'s' if num_micrographs != 1 else ''} with a total of {total_number_of_particles} particles: {time_str}\n")
    
if __name__ == "__main__":
    main()
