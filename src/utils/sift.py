import os
import subprocess
import csv
import pandas as pd
import numpy as np
import nibabel as nib
import tempfile
from tqdm import tqdm


class SIFT3D():
    """
    Usage: kpSift3D [image.nii] 

    Detects SIFT3D keypoints and extracts their descriptors from an image.

    ```
    Example: 
    kpSift3D --keys keys.csv --desc desc.csv image.nii 
    ````

    Output options: 
    --keys [filename] 
        Specifies the output file name for the keypoints. 
        Supported file formats: .csv, .csv.gz 
    --desc [filename] 
        Specifies the output file name for the descriptors. 
        Supported file formats: .csv, .csv.gz 
    --draw [filename] 
        Draws the keypoints in image space. 
        Supported file formats: .dcm, .nii, .nii.gz, directory 
    At least one of the output options must be specified. 


    SIFT3D Options: 
    --peak_thresh [value] 
        The smallest allowed absolute DoG value, as a fraction 
            of the largest. Must be on the interval (0, 1]. 
            (default: 0.10) 
    --corner_thresh [value] 
        The smallest allowed corner score, on the interval 
            [0, 1]. (default: 0.40) 
    --num_kp_levels [value] 
        The number of pyramid levels per octave in which 
            keypoints are found. Must be a positive integer. 
            (default: 3) 
    --sigma_n [value] 
        The nominal scale parameter of the input data, on the 
            interval (0, inf). (default: 1.15) 
    --sigma0 [value] 
        The scale parameter of the first level of octave 0, on 
            the interval (0, inf). (default: 1.60) 

    REF: https://github.com/bbrister/SIFT3D

    """
    def __init__(self, peak_threshold=0.1, corner_threshold=0.4, num_kp_levels=3, sigma_n=1.15, sigma0=1.6):
        self.executable = './external_libs/SIFT3D/build/bin/kpSift3D'
        self.peak_threshold = peak_threshold
        self.corner_threshold = corner_threshold
        self.num_kp_levels = num_kp_levels
        self.sigma_n = sigma_n
        self.sigma0 = sigma0

    def preprocess_nifti_for_sift3d(self, input_file, output_file=None):
        """
        Preprocess NIfTI file by setting the IJK to RAS matrix to have diagonal [-1, -1, 1, 1].
        This replicates the Slicer preprocessing step without changing the actual image data.
        
        Args:
            input_file (str): Path to input NIfTI file
            output_file (str, optional): Path to output file. If None, uses temporary file.
            
        Returns:
            str: Path to the preprocessed file
        """
        # Load the NIfTI image
        img = nib.load(input_file)
        
        # Create the custom affine matrix with diagonal [-1, -1, 1, 1]
        custom_affine = np.diag([-1, -1, 1, 1]).astype(np.float64)
        
        # Create a new image with the same data but modified affine matrix
        preprocessed_img = nib.Nifti1Image(img.get_fdata(), custom_affine, img.header)
        
        # If no output file specified, create a temporary file
        if output_file is None:
            temp_fd, output_file = tempfile.mkstemp(suffix='.nii.gz', prefix='sift3d_preprocessed_')
            os.close(temp_fd)  # Close the file descriptor, we just need the filename
        
        # Save the preprocessed image
        nib.save(preprocessed_img, output_file)
        
        return output_file

    def run_sift3d(self, image_file, output_directory, preprocess=True, cleanup_temp=True):
        """
        Run SIFT3D on an image file with optional preprocessing.
        
        Args:
            image_file (str): Path to input image file
            output_directory (str): Directory to save outputs
            preprocess (bool): Whether to apply IJK to RAS matrix preprocessing
            cleanup_temp (bool): Whether to clean up temporary preprocessed files
        """
        # Preprocess the image if requested
        processed_image_file = image_file
        temp_file_created = False
        
        if preprocess:
            processed_image_file = self.preprocess_nifti_for_sift3d(image_file)
            temp_file_created = True
        
        image_name = os.path.basename(image_file)
        base_name = image_name.split('.nii')[0]

        # Define the output files for keypoints
        keys_output = os.path.join(output_directory, f"{base_name}_keys.csv")
        desc_output = os.path.join(output_directory, f"{base_name}_desc.csv")
        file_name = os.path.join(output_directory, f"{base_name}_keys.nii.gz")

        # Run the SIFT3D command on the processed image
        command = ['sudo',
                    self.executable, 
                #    '--keys', keys_output,
                   '--desc', desc_output, 
                #    '--draw', file_name,
                   '--peak_thresh', str(self.peak_threshold), 
                   '--corner_thresh', str(self.corner_threshold), 
                   '--num_kp_levels', str(self.num_kp_levels),
                   '--sigma_n', str(self.sigma_n), 
                   '--sigma0', str(self.sigma0), 
                   processed_image_file]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Clean up temporary file if it was created
        if temp_file_created and cleanup_temp:
            try:
                os.remove(processed_image_file)
            except OSError:
                pass
        
        # Check if the command was successful
        if result.returncode != 0:
            return 0, None
        
        # Count the number of detected points
        if os.path.exists(desc_output):
            num_points = self.count_keypoints(desc_output)
            return num_points, desc_output
        else:
            return 0, None

    def count_keypoints(self, keypoints_file):
        with open(keypoints_file, 'r') as f:
            csv_reader = csv.reader(f)
            return sum(1 for row in csv_reader)

    def process_images(self, image_files, output_directory, preprocess=True):
        """
        Process multiple images with SIFT3D.
        
        Args:
            image_files (list): List of image file paths
            output_directory (str): Directory to save outputs
            preprocess (bool): Whether to apply IJK to RAS matrix preprocessing
        """
        os.makedirs(output_directory, exist_ok=True)
        total_points = 0
        failed_files = 0
        
        # Use tqdm for progress bar
        for image_file in tqdm(image_files, desc="Processing images", unit="file"):
            points, filename = self.run_sift3d(image_file, output_directory, preprocess=preprocess)
            if filename:
                total_points += points
            else:
                failed_files += 1
        
        # Summary
        success_count = len(image_files) - failed_files
        print(f"✅ Processed {success_count}/{len(image_files)} files successfully")
        print(f"🔑 Total keypoints detected: {total_points}")
        if failed_files > 0:
            print(f"❌ Failed files: {failed_files}")
        
        return total_points


def get_sift_descriptors(image_file, keypoints_file, desc_output):
    
    executable = './external_libs/SIFT3D/build/bin/featSift3D'
    command = ['sudo',
                executable,
                image_file,
                keypoints_file, 
                desc_output]
    subprocess.run(command)
    
    # Unzip the descriptors
    df = pd.read_csv(desc_output, compression='gzip')
    desc_data = df.iloc[:, 3:].values
    desc_data = np.array(desc_data, dtype=np.float32)
    return desc_data