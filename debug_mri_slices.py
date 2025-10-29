"""
Debug script to visualize MRI slices in different planes
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from alignment import normalize_intensity

# Find the latest output directory
OUTPUT_DIR = Path(__file__).parent / "outputs"
session_dirs = sorted(OUTPUT_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

if not session_dirs:
    print("No session found! Please run alignment first.")
    exit(1)

latest_session = session_dirs[0]
aligned_mri_path = latest_session / "aligned_mri.nii.gz"

if not aligned_mri_path.exists():
    print(f"Aligned MRI not found in {latest_session}")
    exit(1)

print(f"Loading: {aligned_mri_path}")

# Load MRI with RAS orientation
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation

mri_img = nib.load(str(aligned_mri_path))
mri_data = mri_img.get_fdata().astype(np.float32)

print(f"Original orientation: {nib.aff2axcodes(mri_img.affine)}")
print(f"Original shape: {mri_data.shape}")

# Reorient to RAS
current_ornt = io_orientation(mri_img.affine)
target_ornt = axcodes2ornt(('R', 'A', 'S'))
transform = ornt_transform(current_ornt, target_ornt)
mri_data_ras = apply_orientation(mri_data, transform)

print(f"RAS shape: {mri_data_ras.shape}")

mri_norm = normalize_intensity(mri_data_ras)

# Get voxel spacing (mm) along R, A, S
sx, sy, sz = mri_img.header.get_zooms()[:3]
print(f"Voxel spacing: sx={sx:.2f}, sy={sy:.2f}, sz={sz:.2f} mm")

# Get middle slices
z_mid = mri_norm.shape[2] // 2
y_mid = mri_norm.shape[1] // 2
x_mid = mri_norm.shape[0] // 2

# Helper functions with proper RAS orientation
def axial_slice(data_ras, z):
    """Axial slice: (R, A) plane, transpose to put R→right, A→up"""
    sl = data_ras[:, :, z]
    extent = [0, data_ras.shape[0]*sx, 0, data_ras.shape[1]*sy]
    return sl.T, extent

def coronal_slice(data_ras, y):
    """Coronal slice: (R, S) plane, transpose to put R→right, S→up"""
    sl = data_ras[:, y, :]
    extent = [0, data_ras.shape[0]*sx, 0, data_ras.shape[2]*sz]
    return sl.T, extent

def sagittal_slice(data_ras, x):
    """Sagittal slice: (A, S) plane, transpose to put A→right, S→up"""
    sl = data_ras[x, :, :]
    extent = [0, data_ras.shape[1]*sy, 0, data_ras.shape[2]*sz]
    return sl.T, extent

# Extract slices with proper orientation
axial, axial_ext = axial_slice(mri_norm, z_mid)
coronal, coronal_ext = coronal_slice(mri_norm, y_mid)
sagittal, sagittal_ext = sagittal_slice(mri_norm, x_mid)

print(f"Axial slice shape: {axial.shape}")
print(f"Coronal slice shape: {coronal.shape}")
print(f"Sagittal slice shape: {sagittal.shape}")

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(axial, cmap='gray', extent=axial_ext, origin='lower', interpolation='nearest')
axes[0].set_title(f'Axial [:, :, {z_mid}]\nR→right, A→up\nShape: {axial.shape}', fontsize=12)
axes[0].set_aspect('equal')
axes[0].axis('off')

axes[1].imshow(coronal, cmap='gray', extent=coronal_ext, origin='lower', interpolation='nearest')
axes[1].set_title(f'Coronal [:, {y_mid}, :]\nR→right, S→up\nShape: {coronal.shape}', fontsize=12)
axes[1].set_aspect('equal')
axes[1].axis('off')

axes[2].imshow(sagittal, cmap='gray', extent=sagittal_ext, origin='lower', interpolation='nearest')
axes[2].set_title(f'Sagittal [{x_mid}, :, :]\nA→right, S→up\nShape: {sagittal.shape}', fontsize=12)
axes[2].set_aspect('equal')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('debug_mri_slices.png', dpi=150, bbox_inches='tight')
print("\nSaved to: debug_mri_slices.png")
plt.show()
