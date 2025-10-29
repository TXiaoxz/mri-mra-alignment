"""
MRIMRA Web Application - FastAPI Backend
Author: Xupeng Zhang
Description: Web interface for MRI-MRA alignment and visualization
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from alignment import align_mri_mra, generate_visualization

# Initialize FastAPI app
app = FastAPI(
    title="MRIMRA Web Application",
    description="Brain MRI-MRA Alignment and Visualization",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Sample data paths
SAMPLE_DATA_DIR = BASE_DIR.parent / "100_Guys" / "scans"
SAMPLE_MRI = SAMPLE_DATA_DIR / "T1-T1" / "resources" / "NIfTI" / "files" / "IXI100-Guys-0747-T1.nii.gz"
SAMPLE_MRA = SAMPLE_DATA_DIR / "MRA-MRA" / "resources" / "NIfTI" / "files" / "IXI100-Guys-0747-MRA.nii.gz"

# Create directories
for directory in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR]:
    directory.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    html_path = TEMPLATES_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "MRIMRA Web Application API"}


@app.post("/api/upload")
async def upload_files(
    mri_file: UploadFile = File(..., description="MRI T1-weighted NIfTI file (.nii.gz)"),
    mra_file: UploadFile = File(..., description="MRA NIfTI file (.nii.gz)")
):
    """
    Upload MRI and MRA files for alignment processing
    """
    # Validate file extensions
    if not mri_file.filename.endswith(('.nii.gz', '.nii')):
        raise HTTPException(status_code=400, detail="MRI file must be .nii or .nii.gz format")

    if not mra_file.filename.endswith(('.nii.gz', '.nii')):
        raise HTTPException(status_code=400, detail="MRA file must be .nii or .nii.gz format")

    # Generate unique session ID
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Save uploaded files
    mri_path = session_dir / f"mri_{mri_file.filename}"
    mra_path = session_dir / f"mra_{mra_file.filename}"

    try:
        # Write files
        with open(mri_path, "wb") as f:
            shutil.copyfileobj(mri_file.file, f)

        with open(mra_path, "wb") as f:
            shutil.copyfileobj(mra_file.file, f)

        return JSONResponse({
            "status": "success",
            "message": "Files uploaded successfully",
            "session_id": session_id,
            "mri_filename": mri_file.filename,
            "mra_filename": mra_file.filename
        })

    except Exception as e:
        # Cleanup on error
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=f"Error saving files: {str(e)}")


@app.post("/api/align/{session_id}")
async def align_images(session_id: str):
    """
    Perform MRI-MRA alignment for a given session
    """
    session_dir = UPLOAD_DIR / session_id

    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Find MRI and MRA files
    mri_files = list(session_dir.glob("mri_*"))
    mra_files = list(session_dir.glob("mra_*"))

    if not mri_files or not mra_files:
        raise HTTPException(status_code=400, detail="MRI or MRA file not found")

    mri_path = str(mri_files[0])
    mra_path = str(mra_files[0])

    # Output directory for this session
    output_dir = OUTPUT_DIR / session_id
    output_dir.mkdir(exist_ok=True)

    try:
        # Perform alignment
        result = align_mri_mra(
            mri_path=mri_path,
            mra_path=mra_path,
            output_dir=str(output_dir)
        )

        # Generate visualizations
        viz_paths = generate_visualization(
            aligned_mri_path=result["aligned_mri"],
            mra_path=mra_path,
            output_dir=str(output_dir)
        )

        return JSONResponse({
            "status": "success",
            "message": "Alignment completed successfully",
            "session_id": session_id,
            "results": {
                "aligned_mri": result["aligned_mri"],
                "transform_parameters": result.get("transform_params", {}),
                "visualizations": viz_paths
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alignment error: {str(e)}")


@app.get("/api/result/{session_id}/{image_type}")
async def get_result_image(session_id: str, image_type: str):
    """
    Retrieve result images (mri_axial, mra_axial, overlay, etc.)
    """
    output_dir = OUTPUT_DIR / session_id

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Session results not found")

    # Map image types to file names
    image_map = {
        "mri_original": "mri_axial.png",
        "mra_original": "mra_axial.png",
        "aligned_mri": "aligned_mri_axial.png",
        "overlay": "overlay.png",
        "vessel_mask": "vessel_mask.png"
    }

    if image_type not in image_map:
        raise HTTPException(status_code=400, detail="Invalid image type")

    image_path = output_dir / image_map[image_type]

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {image_type} not found")

    return FileResponse(image_path, media_type="image/png")


@app.get("/api/slice/{session_id}/{slice_index}")
async def get_slice_image(session_id: str, slice_index: int):
    """
    Get a specific slice for both aligned MRI and MRA
    Returns PNG image with MRI on left, MRA on right
    """
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    from alignment import normalize_intensity, normalize_uint8

    session_upload_dir = UPLOAD_DIR / session_id
    session_output_dir = OUTPUT_DIR / session_id

    if not session_upload_dir.exists() or not session_output_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Load aligned MRI and original MRA
    aligned_mri_path = session_output_dir / "aligned_mri.nii.gz"
    mra_files = list(session_upload_dir.glob("mra_*"))

    if not aligned_mri_path.exists() or not mra_files:
        raise HTTPException(status_code=404, detail="Aligned images not found")

    mra_path = str(mra_files[0])

    # Load data
    mri_img = nib.load(str(aligned_mri_path))
    mra_img = nib.load(mra_path)

    mri_data = mri_img.get_fdata()
    mra_data = mra_img.get_fdata()

    # Normalize
    mri_norm = normalize_intensity(mri_data.astype(np.float32))
    mra_norm = normalize_intensity(mra_data.astype(np.float32))

    # After registration, both MRI and MRA are in the same space (RAS coordinate system)
    # So we use the same slicing method for both: [:, :, z] for axial view
    # Transpose to put R→right, A→up (proper radiological view)
    max_slices = min(mri_norm.shape[2], mra_norm.shape[2])
    slice_index = min(slice_index, max_slices - 1)

    mri_slice = mri_norm[:, :, slice_index].T
    mra_slice = mra_norm[:, :, slice_index].T

    # Get pixel spacing for correct aspect ratio
    # Both are now in the same space (MRA space), so use MRA spacing
    mra_spacing = mra_img.header.get_zooms()

    # For axial slice in RAS after transpose: aspect = sy / sx
    aspect_ratio = mra_spacing[1] / mra_spacing[0]

    # Create side-by-side comparison image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(mri_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    axes[0].set_title(f'Aligned MRI - Slice {slice_index}', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(mra_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
    axes[1].set_title(f'MRA - Slice {slice_index}', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/pregenerate-slices/{session_id}")
async def pregenerate_slices(session_id: str, step: int = 1):
    """
    Pre-generate all slice images for fast navigation
    step: generate every Nth slice (1 = all slices, 2 = every other slice, etc.)
    """
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from alignment import normalize_intensity

    session_upload_dir = UPLOAD_DIR / session_id
    session_output_dir = OUTPUT_DIR / session_id

    if not session_upload_dir.exists() or not session_output_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Create slices directory
    slices_dir = session_output_dir / "slices"
    slices_dir.mkdir(exist_ok=True)

    # Load aligned MRI and MRA
    aligned_mri_path = session_output_dir / "aligned_mri.nii.gz"
    mra_files = list(session_upload_dir.glob("mra_*"))

    if not aligned_mri_path.exists() or not mra_files:
        raise HTTPException(status_code=404, detail="Aligned images not found")

    mri_img = nib.load(str(aligned_mri_path))
    mra_img = nib.load(str(mra_files[0]))

    mri_data = mri_img.get_fdata()
    mra_data = mra_img.get_fdata()

    mri_norm = normalize_intensity(mri_data.astype(np.float32))
    mra_norm = normalize_intensity(mra_data.astype(np.float32))

    max_slices = min(mri_norm.shape[2], mra_norm.shape[2])
    mra_spacing = mra_img.header.get_zooms()
    aspect_ratio = mra_spacing[1] / mra_spacing[0]

    # Generate slices
    generated = []
    for z in range(0, max_slices, step):
        mri_slice = mri_norm[:, :, z].T
        mra_slice = mra_norm[:, :, z].T

        # Create side-by-side image
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(mri_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
        axes[0].set_title(f'Aligned MRI - Slice {z}', fontsize=14)
        axes[0].axis('off')

        axes[1].imshow(mra_slice, cmap='gray', aspect=aspect_ratio, origin='lower', interpolation='nearest')
        axes[1].set_title(f'MRA - Slice {z}', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()

        slice_path = slices_dir / f"slice_{z:03d}.png"
        plt.savefig(slice_path, dpi=100, bbox_inches='tight')
        plt.close()

        generated.append(z)

    return JSONResponse({
        "status": "success",
        "slices_generated": len(generated),
        "slice_indices": generated,
        "total_slices": max_slices,
        "step": step
    })


@app.get("/api/volume-info/{session_id}")
async def get_volume_info(session_id: str):
    """
    Get volume dimensions for slice navigation
    """
    import nibabel as nib

    session_upload_dir = UPLOAD_DIR / session_id
    session_output_dir = OUTPUT_DIR / session_id

    if not session_output_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    aligned_mri_path = session_output_dir / "aligned_mri.nii.gz"
    mra_files = list(session_upload_dir.glob("mra_*"))

    if not aligned_mri_path.exists() or not mra_files:
        raise HTTPException(status_code=404, detail="Aligned images not found")

    mri_img = nib.load(str(aligned_mri_path))
    mra_img = nib.load(str(mra_files[0]))

    mri_shape = mri_img.shape
    mra_shape = mra_img.shape

    # After registration, both use z dimension for axial view
    num_slices = min(mri_shape[2], mra_shape[2])

    # Check if slices are pre-generated
    slices_dir = session_output_dir / "slices"
    pregenerated = slices_dir.exists() and len(list(slices_dir.glob("slice_*.png"))) > 0

    return JSONResponse({
        "mri_shape": list(mri_shape),
        "mra_shape": list(mra_shape),
        "num_slices": int(num_slices),
        "pregenerated": pregenerated,
        "note": "Both MRI[:, :, z] and MRA[:, :, z] show axial view after registration in RAS"
    })


@app.get("/api/pregenerated-slice/{session_id}/{slice_index}")
async def get_pregenerated_slice(session_id: str, slice_index: int):
    """
    Get a pre-generated slice image
    """
    session_output_dir = OUTPUT_DIR / session_id
    slices_dir = session_output_dir / "slices"

    slice_path = slices_dir / f"slice_{slice_index:03d}.png"

    if not slice_path.exists():
        raise HTTPException(status_code=404, detail=f"Slice {slice_index} not found")

    return FileResponse(slice_path, media_type="image/png")


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete session files and results
    """
    session_upload_dir = UPLOAD_DIR / session_id
    session_output_dir = OUTPUT_DIR / session_id

    deleted = []

    if session_upload_dir.exists():
        shutil.rmtree(session_upload_dir)
        deleted.append("uploads")

    if session_output_dir.exists():
        shutil.rmtree(session_output_dir)
        deleted.append("outputs")

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse({
        "status": "success",
        "message": f"Session {session_id} deleted",
        "deleted": deleted
    })


@app.post("/api/use-sample-data")
async def use_sample_data():
    """
    Use the sample data (IXI100-Guys-0747) for alignment
    """
    # Check if sample data exists
    if not SAMPLE_MRI.exists() or not SAMPLE_MRA.exists():
        raise HTTPException(
            status_code=404,
            detail="Sample data not found. Please ensure the 100_Guys folder exists in the parent directory."
        )

    # Generate session ID
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    try:
        # Copy sample files to session directory
        mri_dest = session_dir / "mri_IXI100-Guys-0747-T1.nii.gz"
        mra_dest = session_dir / "mra_IXI100-Guys-0747-MRA.nii.gz"

        shutil.copy2(SAMPLE_MRI, mri_dest)
        shutil.copy2(SAMPLE_MRA, mra_dest)

        return JSONResponse({
            "status": "success",
            "message": "Sample data loaded successfully",
            "session_id": session_id,
            "mri_filename": "IXI100-Guys-0747-T1.nii.gz",
            "mra_filename": "IXI100-Guys-0747-MRA.nii.gz"
        })

    except Exception as e:
        # Cleanup on error
        if session_dir.exists():
            shutil.rmtree(session_dir)
        raise HTTPException(status_code=500, detail=f"Error loading sample data: {str(e)}")


@app.get("/api/debug-orientations/{session_id}/{slice_index}")
async def debug_orientations(session_id: str, slice_index: int):
    """
    Debug endpoint: show MRI and MRA with different slice planes (axial, coronal, sagittal)
    and different rotations/flips for each plane
    """
    import nibabel as nib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from io import BytesIO
    from alignment import normalize_intensity

    session_upload_dir = UPLOAD_DIR / session_id
    session_output_dir = OUTPUT_DIR / session_id

    if not session_upload_dir.exists() or not session_output_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Load aligned MRI and original MRA
    aligned_mri_path = session_output_dir / "aligned_mri.nii.gz"
    mra_files = list(session_upload_dir.glob("mra_*"))

    if not aligned_mri_path.exists() or not mra_files:
        raise HTTPException(status_code=404, detail="Aligned images not found")

    mra_path = str(mra_files[0])

    # Load data
    mri_img = nib.load(str(aligned_mri_path))
    mra_img = nib.load(mra_path)

    mri_data = mri_img.get_fdata()
    mra_data = mra_img.get_fdata()

    # Normalize
    mri_norm = normalize_intensity(mri_data.astype(np.float32))
    mra_norm = normalize_intensity(mra_data.astype(np.float32))

    print(f"MRI shape: {mri_norm.shape}")
    print(f"MRA shape: {mra_norm.shape}")

    # Define slice planes and how to extract them
    # For each plane, we'll show the slice and multiple rotations
    slice_configs = []

    # Axial plane (z-axis) - top-down view
    z_idx = min(slice_index, mri_norm.shape[2] - 1)
    slice_configs.append({
        'name': 'Axial [:,:,z]',
        'mri': mri_norm[:, :, z_idx],
        'mra': mra_norm[:, :, z_idx]
    })

    # Coronal plane (y-axis) - front-back view
    y_idx = min(slice_index, mri_norm.shape[1] - 1)
    slice_configs.append({
        'name': 'Coronal [:,y,:]',
        'mri': mri_norm[:, y_idx, :],
        'mra': mra_norm[:, y_idx, :]
    })

    # Sagittal plane (x-axis) - left-right view
    x_idx = min(slice_index, mri_norm.shape[0] - 1)
    slice_configs.append({
        'name': 'Sagittal [x,:,:]',
        'mri': mri_norm[x_idx, :, :],
        'mra': mra_norm[x_idx, :, :]
    })

    # Transformations to try
    transforms = {
        'Original': lambda x: x,
        'Rot90': lambda x: np.rot90(x, k=1),
        'Rot180': lambda x: np.rot90(x, k=2),
        'Rot270': lambda x: np.rot90(x, k=3),
    }

    # Create figure: 3 planes x 4 transforms = 12 rows, 2 columns (MRI, MRA)
    n_rows = len(slice_configs) * len(transforms)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows))

    row_idx = 0
    for plane_config in slice_configs:
        for trans_name, trans_func in transforms.items():
            # MRI
            mri_slice = trans_func(plane_config['mri'].copy())
            axes[row_idx, 0].imshow(mri_slice, cmap='gray')
            axes[row_idx, 0].set_title(f"MRI - {plane_config['name']} - {trans_name}", fontsize=11)
            axes[row_idx, 0].axis('off')

            # MRA
            mra_slice = trans_func(plane_config['mra'].copy())
            axes[row_idx, 1].imshow(mra_slice, cmap='gray')
            axes[row_idx, 1].set_title(f"MRA - {plane_config['name']} - {trans_name}", fontsize=11)
            axes[row_idx, 1].axis('off')

            row_idx += 1

    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png")


@app.get("/api/debug-orientations-latest/{slice_index}")
async def debug_orientations_latest(slice_index: int = 50):
    """
    Debug endpoint using the latest session
    """
    # Find the most recent session
    session_dirs = sorted(OUTPUT_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not session_dirs:
        raise HTTPException(status_code=404, detail="No sessions found")

    latest_session_id = session_dirs[0].name
    return await debug_orientations(latest_session_id, slice_index)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MRIMRA Web Application"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
