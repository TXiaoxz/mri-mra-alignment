"""
MRI-MRA Alignment Core Module
Based on the research implementation from mrimra.ipynb
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
from skimage.filters import frangi
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path


def load_nifti(path: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    """
    Load NIfTI file and return data array and image object
    Data is reoriented to RAS (Right-Anterior-Superior) coordinate system

    Args:
        path: Path to .nii or .nii.gz file

    Returns:
        data: 3D numpy array in RAS orientation
        img: nibabel image object (reoriented)
    """
    from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation

    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)

    # Get current orientation
    current_ornt = io_orientation(img.affine)

    # Target orientation: RAS (Right-Anterior-Superior)
    # This is the standard orientation used by 3D Slicer and many tools
    target_ornt = axcodes2ornt(('R', 'A', 'S'))

    # Calculate transformation
    transform = ornt_transform(current_ornt, target_ornt)

    # Apply orientation transformation to data
    data_ras = apply_orientation(data, transform)

    # Create new image with RAS orientation
    img_ras = nib.as_closest_canonical(img)

    print(f"  Original orientation: {nib.aff2axcodes(img.affine)}")
    print(f"  RAS orientation: {nib.aff2axcodes(img_ras.affine)}")
    print(f"  Original shape: {data.shape} -> RAS shape: {data_ras.shape}")

    return data_ras, img_ras


def normalize_intensity(data: np.ndarray, percentile_range: tuple = (1, 99)) -> np.ndarray:
    """
    Normalize intensity using percentile clipping

    Args:
        data: Input array
        percentile_range: (lower, upper) percentiles for clipping

    Returns:
        Normalized array [0, 1]
    """
    lo, hi = np.percentile(data, percentile_range[0]), np.percentile(data, percentile_range[1])
    data_norm = np.clip((data - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return data_norm


def extract_vessels_frangi(mra_data: np.ndarray, sigmas: tuple = (1, 3, 5)) -> np.ndarray:
    """
    Extract vessel structures using Frangi vesselness filter

    Args:
        mra_data: 3D MRA volume
        sigmas: Scale range for vessel detection

    Returns:
        vessel_map: 3D vesselness probability map
    """
    print("Applying Frangi filter for vessel enhancement...")

    # Normalize input
    mra_norm = normalize_intensity(mra_data)

    # Apply Frangi filter on each slice
    vessel_map = np.zeros_like(mra_norm)

    for z in range(mra_norm.shape[2]):
        slice_2d = mra_norm[:, :, z]
        vessel_slice = frangi(slice_2d, sigmas=sigmas, black_ridges=False)
        vessel_map[:, :, z] = vessel_slice

    return vessel_map


def align_mri_mra(
    mri_path: str,
    mra_path: str,
    output_dir: str,
    registration_type: str = "affine"
) -> dict:
    """
    Main alignment function for MRI-MRA registration

    Args:
        mri_path: Path to MRI T1-weighted NIfTI file (moving image)
        mra_path: Path to MRA NIfTI file (fixed image)
        output_dir: Directory to save results
        registration_type: "rigid" or "affine"

    Returns:
        Dictionary containing paths and metadata
    """
    print(f"Starting MRI-MRA alignment...")
    print(f"MRI: {mri_path}")
    print(f"MRA: {mra_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load images
    mri_data, mri_img = load_nifti(mri_path)
    mra_data, mra_img = load_nifti(mra_path)

    print(f"MRI shape: {mri_data.shape}")
    print(f"MRA shape: {mra_data.shape}")

    # Normalize intensities
    mri_norm = normalize_intensity(mri_data)
    mra_norm = normalize_intensity(mra_data)

    # Convert nibabel to SimpleITK with full metadata (direction, origin, spacing)
    # This is critical to preserve orientation information
    def nib_to_sitk(nib_img, data):
        """Convert nibabel image to SimpleITK with full metadata"""
        # SimpleITK expects (z, y, x) order
        sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))

        # Set spacing
        spacing = [float(x) for x in nib_img.header.get_zooms()[:3]]
        sitk_img.SetSpacing(spacing)

        # Set origin
        affine = nib_img.affine
        origin = affine[:3, 3]
        sitk_img.SetOrigin(origin.tolist())

        # Set direction (rotation matrix from affine)
        direction_matrix = affine[:3, :3]
        # Normalize by spacing to get pure rotation
        direction_matrix = direction_matrix / np.array(spacing)[:, None]
        # Flatten in column-major order (Fortran order) for SimpleITK
        direction = direction_matrix.flatten(order='F').tolist()
        sitk_img.SetDirection(direction)

        return sitk_img

    mri_sitk = nib_to_sitk(mri_img, mri_norm)
    mra_sitk = nib_to_sitk(mra_img, mra_norm)

    print(f"MRI SimpleITK:")
    print(f"  Origin: {mri_sitk.GetOrigin()}")
    print(f"  Direction: {mri_sitk.GetDirection()}")
    print(f"MRA SimpleITK:")
    print(f"  Origin: {mra_sitk.GetOrigin()}")
    print(f"  Direction: {mra_sitk.GetDirection()}")

    # Initialize registration
    print("Initializing registration...")
    registration = sitk.ImageRegistrationMethod()

    # Metric: Mattes Mutual Information
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)

    # Optimizer
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)

    # Set transform type
    if registration_type == "rigid":
        initial_transform = sitk.CenteredTransformInitializer(
            mra_sitk, mri_sitk, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    else:  # affine
        initial_transform = sitk.CenteredTransformInitializer(
            mra_sitk, mri_sitk, sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

    registration.SetInitialTransform(initial_transform)

    # Multi-resolution framework
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Execute registration
    print(f"Performing {registration_type} registration...")
    final_transform = registration.Execute(mra_sitk, mri_sitk)

    print(f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
    print(f"Final metric value: {registration.GetMetricValue()}")

    # Apply transform
    mri_aligned_sitk = sitk.Resample(
        mri_sitk,
        mra_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        mri_sitk.GetPixelID()
    )

    # Convert back to numpy
    mri_aligned = np.transpose(sitk.GetArrayFromImage(mri_aligned_sitk), (2, 1, 0))

    print(f"After registration:")
    print(f"  Aligned MRI shape: {mri_aligned.shape}")
    print(f"  MRA shape: {mra_data.shape}")

    # Save aligned MRI
    aligned_nifti_path = output_path / "aligned_mri.nii.gz"
    aligned_img = nib.Nifti1Image(mri_aligned, mra_img.affine, mra_img.header)
    nib.save(aligned_img, aligned_nifti_path)

    print(f"Aligned MRI saved to: {aligned_nifti_path}")

    # Extract transform parameters
    transform_params = {
        "transform_type": registration_type,
        "metric_value": float(registration.GetMetricValue()),
        "iterations": registration.GetOptimizerIteration()
    }

    return {
        "aligned_mri": str(aligned_nifti_path),
        "transform_params": transform_params,
        "mri_aligned_data": mri_aligned,
        "mra_data": mra_norm
    }


def normalize_uint8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 range [0, 255]"""
    img_norm = (img - np.min(img)) / (np.ptp(img) + 1e-8)
    return (img_norm * 255).astype(np.uint8)


def generate_visualization(
    aligned_mri_path: str,
    mra_path: str,
    output_dir: str,
    slice_index: int = None
) -> dict:
    """
    Generate visualization images for web display

    Args:
        aligned_mri_path: Path to aligned MRI
        mra_path: Path to original MRA
        output_dir: Output directory for images
        slice_index: Specific slice index (None for middle)

    Returns:
        Dictionary with paths to generated images
    """
    print("Generating visualizations...")

    output_path = Path(output_dir)

    # Load data
    mri_aligned, mri_img = load_nifti(aligned_mri_path)
    mra_data, mra_img = load_nifti(mra_path)

    # Normalize
    mri_norm = normalize_intensity(mri_aligned)
    mra_norm = normalize_intensity(mra_data)

    # Select slice
    # After registration, aligned MRI is in MRA space, so use same slicing method
    if slice_index is None:
        slice_index = mri_norm.shape[2] // 2  # Both use z-axis after registration

    # Extract axial slices - both use the same indexing after registration
    # In RAS coordinate system: axial view is [:, :, z] from superior to inferior
    # Transpose to put R→right, A→up (proper radiological view)
    mri_slice = mri_norm[:, :, slice_index].T
    mra_slice = mra_norm[:, :, slice_index].T

    # Get pixel spacing for correct aspect ratio
    # After registration, both are in MRA space, so use MRA spacing
    mra_spacing = mra_img.header.get_zooms()

    # For axial slice in RAS after transpose: aspect = sy / sx
    aspect_ratio = mra_spacing[1] / mra_spacing[0]

    # Create vessel map using Frangi filter
    print("Creating vessel map...")
    vessel_map = frangi(mra_slice, sigmas=(1, 3, 5), black_ridges=False)

    # Save individual images
    viz_paths = {}

    # 1. Aligned MRI axial
    plt.figure(figsize=(8, 8))
    plt.imshow(mri_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    plt.title('Aligned MRI (Axial) - RAS')
    plt.axis('off')
    plt.tight_layout()
    mri_path = output_path / "aligned_mri_axial.png"
    plt.savefig(mri_path, dpi=150, bbox_inches='tight')
    plt.close()
    viz_paths['aligned_mri'] = str(mri_path)

    # 2. MRA axial
    plt.figure(figsize=(8, 8))
    plt.imshow(mra_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    plt.title('MRA (Axial) - RAS')
    plt.axis('off')
    plt.tight_layout()
    mra_path = output_path / "mra_axial.png"
    plt.savefig(mra_path, dpi=150, bbox_inches='tight')
    plt.close()
    viz_paths['mra'] = str(mra_path)

    # 3. Vessel mask
    plt.figure(figsize=(8, 8))
    plt.imshow(vessel_map, cmap='hot', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    plt.title('Vessel Map (Frangi Filter)')
    plt.axis('off')
    plt.tight_layout()
    vessel_path = output_path / "vessel_mask.png"
    plt.savefig(vessel_path, dpi=150, bbox_inches='tight')
    plt.close()
    viz_paths['vessel_mask'] = str(vessel_path)

    # 4. Overlay: MRI (gray) + Vessels (color)
    plt.figure(figsize=(8, 8))
    plt.imshow(mri_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    plt.imshow(vessel_map, cmap='hot', alpha=0.4, aspect=aspect_ratio, origin='lower', interpolation='nearest')
    plt.title('Overlay: MRI + Vessels')
    plt.axis('off')
    plt.tight_layout()
    overlay_path = output_path / "overlay.png"
    plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
    plt.close()
    viz_paths['overlay'] = str(overlay_path)

    # 5. Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(mri_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    axes[0].set_title('Aligned MRI (Axial) - RAS', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(mra_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    axes[1].set_title('MRA (Axial) - RAS', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(mri_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    axes[2].imshow(vessel_map, cmap='hot', alpha=0.4, aspect=aspect_ratio, origin='lower', interpolation='nearest')
    axes[2].set_title('Overlay: MRI + Vessels', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    comparison_path = output_path / "comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    viz_paths['comparison'] = str(comparison_path)

    print(f"Visualizations saved to: {output_dir}")

    return viz_paths


if __name__ == "__main__":
    # Test with sample data
    import sys

    if len(sys.argv) >= 3:
        mri_path = sys.argv[1]
        mra_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "./test_output"

        result = align_mri_mra(mri_path, mra_path, output_dir)
        print("\nAlignment completed!")
        print(f"Results: {result}")

        viz = generate_visualization(
            result['aligned_mri'],
            mra_path,
            output_dir
        )
        print(f"\nVisualizations: {viz}")
    else:
        print("Usage: python alignment.py <mri_path> <mra_path> [output_dir]")
