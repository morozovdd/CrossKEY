from typing import Union
from pathlib import Path
import logging
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)

# Standard affine: 1mm isotropic, origin at 0, LPS orientation (matches real US)
STANDARD_AFFINE = np.diag([-1.0, -1.0, 1.0, 1.0])


def load_nifti(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load NIfTI file using nibabel and transpose to [H, W, D] format.

    NIfTI get_fdata() returns data in [X, Y, Z] anatomical format, but PyTorch
    expects [H, W, D] tensor format. This function handles the coordinate
    transformation to ensure consistency with the dataset expectations.

    Args:
        file_path: Path to NIfTI file

    Returns:
        Volume data as numpy array in [H, W, D] format (transposed from [X, Y, Z])
    """
    img = nib.load(str(file_path))
    data = img.get_fdata()  # Returns [X, Y, Z]
    return data.transpose(1, 0, 2)  # Convert to [Y, X, Z] = [H, W, D]


def standardize_nifti(file_path: Union[str, Path], output_path: Union[str, Path] = None) -> Path:
    """
    Standardize a NIfTI file to have 1mm isotropic spacing, zero origin,
    and standard LPS orientation (affine = diag(-1, -1, 1, 1)).

    The voxel data is preserved as-is; only the header/affine is overwritten.
    This is necessary for SIFT3D, which uses header metadata for scale-space
    computation.

    Args:
        file_path: Path to the input NIfTI file.
        output_path: Where to save the standardized file. If None, overwrites
                     the input file.

    Returns:
        Path to the saved standardized file.
    """
    file_path = Path(file_path)
    if output_path is None:
        output_path = file_path
    output_path = Path(output_path)

    img = nib.load(str(file_path))
    data = img.get_fdata()

    new_img = nib.Nifti1Image(data.astype(np.float32), STANDARD_AFFINE)
    nib.save(new_img, str(output_path))
    logger.info(f"Standardized {file_path.name} -> {output_path.name} "
                f"(shape={data.shape}, affine=diag(-1,-1,1,1), spacing=1mm)")
    return output_path


def save_nifti(data: np.ndarray, file_path: Union[str, Path], reference_path: Union[str, Path] = None) -> None:
    """
    Save numpy array as NIfTI file.
    
    Handles coordinate transformation from PyTorch [H, W, D] format back to 
    NIfTI [X, Y, Z] anatomical format for proper spatial orientation.
    
    Args:
        data: Volume data as numpy array in [H, W, D] format
        file_path: Output path for NIfTI file
        reference_path: Optional reference NIfTI file to copy header/affine from
    """
    # Convert from [H, W, D] back to [X, Y, Z] for NIfTI
    data_transposed = data.transpose(1, 0, 2)  # [H, W, D] -> [X, Y, Z]
    
    if reference_path is not None:
        # Load reference image to get header and affine information
        ref_img = nib.load(str(reference_path))
        # Create new image with same affine and header
        new_img = nib.Nifti1Image(data_transposed, ref_img.affine, ref_img.header)
    else:
        # Create simple NIfTI image with identity affine
        new_img = nib.Nifti1Image(data_transposed, np.eye(4))
    
    nib.save(new_img, str(file_path))


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Normalize volume using min-max normalization.
    
    Args:
        volume: Input volume array
        
    Returns:
        Normalized volume with values in range [0, 1]
    """
    volume_min = volume.min()
    volume_max = volume.max()
    
    if volume_max == volume_min:
        return np.zeros_like(volume)
    
    return (volume - volume_min) / (volume_max - volume_min)