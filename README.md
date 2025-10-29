# MRI-MRA Alignment Web Application

Web-based medical image registration tool for automatic Brain MRI-MRA alignment with interactive slice visualization.

## Features

- Upload MRI (T1-weighted) and MRA NIfTI files
- Automatic multi-modal image registration (rigid/affine)
- **RAS coordinate system standardization** for consistent orientation
- **Interactive slice viewer** with real-time navigation
- Vessel enhancement using Frangi filter
- Real-time progress tracking
- Side-by-side comparison visualization
- Overlay views (MRI structure + vessels)

## Tech Stack

**Backend**: FastAPI, SimpleITK, nibabel, scikit-image, matplotlib
**Frontend**: HTML5, CSS3, Vanilla JavaScript
**Image Processing**: RAS orientation correction, intensity normalization, affine transformation

## Quick Start

### Installation

```bash
cd /Users/dlwl/JHU/Research/MRIMRA/webapp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Server

```bash
python app.py
```

Or using uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open browser: `http://localhost:8000`

## Usage

### Option 1: Use Sample Data
1. Click **"Use Sample Data"** to load IXI100-Guys-0747 test dataset
2. Automatic alignment will start
3. Navigate results using interactive slice viewer

### Option 2: Upload Your Own Data
1. **Upload Files**: Select MRI (.nii/.nii.gz) and MRA files
2. **Process**: Click "Upload & Process" button
3. **View Results**:
   - Interactive slice viewer (drag slider to navigate through slices)
   - Side-by-side comparison: Aligned MRI vs MRA
   - Static results: vessel map (Frangi filter) and overlay visualization
   - Registration metadata (transform type, metric value, iterations)

## API Endpoints

### Use Sample Data
```bash
POST /api/use-sample-data
```

### Upload Files
```bash
POST /api/upload
Content-Type: multipart/form-data
```

### Perform Alignment
```bash
POST /api/align/{session_id}?registration_type=affine
```

### Get Volume Info
```bash
GET /api/volume-info/{session_id}
```

### Get Interactive Slice
```bash
GET /api/slice/{session_id}/{slice_index}
```

### Get Result Image
```bash
GET /api/result/{session_id}/{image_type}
# image_type: aligned_mri, mra_original, vessel_mask, overlay
```

### Delete Session
```bash
DELETE /api/session/{session_id}
```

### Health Check
```bash
GET /api/health
```

## Project Structure

```
webapp/
├── app.py                    # FastAPI application
├── alignment.py              # Core registration algorithm
├── requirements.txt          # Dependencies
├── README.md                # Documentation
├── templates/
│   └── index.html           # Web interface
├── sample_data/             # Sample NIfTI files
├── uploads/                 # Temporary file storage
└── outputs/                 # Processing results (per session)
```

## Algorithm

### 1. Image Loading & Orientation
- Load NIfTI files using nibabel
- **RAS orientation correction**: Convert all images to RAS (Right-Anterior-Superior) coordinate system
- Ensures consistent anatomical orientation across datasets

### 2. Preprocessing
- Intensity normalization (1-99 percentile clipping)
- Metadata preservation (spacing, origin, direction matrix)

### 3. Registration
- **Method**: Mattes Mutual Information
- **Optimizer**: Gradient Descent with automatic scaling
- **Transform**: Affine (default) or Rigid
- **Multi-resolution**: 3 levels (shrink factors: 4, 2, 1)
- **Complete metadata transfer**: spacing, origin, and direction matrix preserved

### 4. Vessel Enhancement
- Multi-scale Frangi vesselness filter
- Scales (sigmas): 1, 3, 5 pixels
- Detects tubular structures (blood vessels)

### 5. Visualization
- Axial slices in RAS coordinate system
- Transposed for proper radiological view (R→right, A→up)
- Correct aspect ratio based on voxel spacing
- Interactive navigation through all slices

## Key Technical Features

### RAS Coordinate System
All images are automatically reoriented to RAS (Right-Anterior-Superior):
- **R (Right)**: X-axis points to patient's right
- **A (Anterior)**: Y-axis points to patient's front
- **S (Superior)**: Z-axis points to patient's head

This ensures consistent orientation regardless of input image format.

### Metadata Preservation
The registration process preserves complete spatial metadata:
- **Spacing**: Voxel dimensions (mm)
- **Origin**: Image position in world coordinates
- **Direction**: Orientation matrix (rotation)

This ensures accurate alignment and proper anatomical correspondence.

### Interactive Slice Viewer
- Navigate through 100 slices using slider
- Side-by-side comparison of aligned MRI and MRA
- Real-time slice generation
- Proper aspect ratio and orientation

## Testing

### Test with sample data:

```bash
python alignment.py \
  sample_data/mri_IXI100-Guys-0747-T1.nii.gz \
  sample_data/mra_IXI100-Guys-0747-MRA.nii.gz \
  ./test_output
```

### Debug slice orientations:

```bash
python debug_mri_slices.py
```

This generates visualization showing axial, coronal, and sagittal views with proper RAS orientation.

## Configuration

Edit parameters in `alignment.py`:

```python
# Registration type
registration_type = "affine"  # or "rigid"

# Optimizer settings
learningRate = 1.0
numberOfIterations = 100
convergenceMinimumValue = 1e-6

# Frangi filter settings
sigmas = (1, 3, 5)  # Multi-scale vessel detection

# Visualization settings
percentile_range = (1, 99)  # Intensity normalization
```

## Troubleshooting

### Port Already in Use
```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Server Auto-reload
The server uses `--reload` mode and automatically reloads when you modify:
- `app.py`
- `alignment.py`
- `templates/index.html`

### Image Orientation Issues
If images appear rotated or flipped:
1. Check that RAS orientation is being applied in `load_nifti()`
2. Verify transpose operation in visualization code
3. Use `debug_mri_slices.py` to inspect orientation

## Author

**Xupeng Zhang**
Johns Hopkins University
M.S. in Electrical and Computer Engineering

## Acknowledgments

- Sample data from IXI Dataset (IXI100-Guys-0747)
- Built with SimpleITK, nibabel, and scikit-image
