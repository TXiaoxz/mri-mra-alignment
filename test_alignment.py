"""
Test script to verify MRI-MRA alignment functionality
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from alignment import align_mri_mra, generate_visualization

# Sample data paths
MRI_PATH = "../100_Guys/scans/T1-T1/resources/NIfTI/files/IXI100-Guys-0747-T1.nii.gz"
MRA_PATH = "../100_Guys/scans/MRA-MRA/resources/NIfTI/files/IXI100-Guys-0747-MRA.nii.gz"
OUTPUT_DIR = "./test_output"

def test_alignment():
    print("=" * 60)
    print("Testing MRI-MRA Alignment")
    print("=" * 60)

    print(f"\nMRI Path: {MRI_PATH}")
    print(f"MRA Path: {MRA_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")

    try:
        # Test alignment
        print("\n" + "=" * 60)
        print("Step 1: Running alignment...")
        print("=" * 60)

        result = align_mri_mra(
            mri_path=MRI_PATH,
            mra_path=MRA_PATH,
            output_dir=OUTPUT_DIR,
            registration_type="affine"
        )

        print(f"\n✓ Alignment completed successfully!")
        print(f"  - Aligned MRI saved to: {result['aligned_mri']}")
        print(f"  - Transform type: {result['transform_params']['transform_type']}")
        print(f"  - Metric value: {result['transform_params']['metric_value']:.6f}")
        print(f"  - Iterations: {result['transform_params']['iterations']}")

        # Test visualization
        print("\n" + "=" * 60)
        print("Step 2: Generating visualizations...")
        print("=" * 60)

        viz_paths = generate_visualization(
            aligned_mri_path=result['aligned_mri'],
            mra_path=MRA_PATH,
            output_dir=OUTPUT_DIR
        )

        print(f"\n✓ Visualizations generated successfully!")
        for key, path in viz_paths.items():
            print(f"  - {key}: {path}")

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_alignment()
    sys.exit(0 if success else 1)
