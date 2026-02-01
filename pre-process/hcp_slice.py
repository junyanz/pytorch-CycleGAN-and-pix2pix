# Slices the 7T scans from the HCP dataset

# For each scan type, extracts max slices from axial, coronal and sagittal directions

# Saves files to specified directory in sub directory for direction file name format: <subjectID>_<scanType>_<sliceDirection>_<sliceNumber>

import argparse
import numpy as np
from pathlib import Path
import nibabel as nib
import zipfile
from PIL import Image
import os
import tempfile





def slice_scan(scan_data):
    slices = []

    # Axial
    for i in range(scan_data.shape[2]):
        slices.append((scan_data[:, :, i], 'axial', i))
    # Coronal
    for i in range(scan_data.shape[1]):
        slices.append((scan_data[:, i, :], 'coronal', i))
    # # Sagittal
    for i in range(scan_data.shape[0]):
        slices.append((scan_data[i, :, :], 'sagittal', i))
    
    return slices


def load_scan(path, subject_id):
    internal_scan_path = f"{subject_id}/T1w/T1w_acpc_dc_restore.nii.gz"

    # Open the zip file
    with zipfile.ZipFile(path, 'r') as z:

        # Open internal scan file
        with z.open(internal_scan_path) as f:
            
            # Write to temp file
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(f.read())                
                tmp.flush()
                
                tmp_path = tmp.name

    img = nib.load(tmp_path)

    return img.get_fdata(), tmp_path



def save_slice(slice_tuple, save_path, subject_id, scan_type):

    slice_array, direction, slice_number = slice_tuple

    slice_array = slice_array.astype(np.float32)
    slice_array -= slice_array.min()
    max_val = slice_array.max()
    if max_val > 0:
        slice_array /= max_val
    slice_array = (slice_array * 255).astype(np.uint8)

    save_path = Path(save_path) / direction
    save_path.mkdir(parents=True, exist_ok=True)

    filename = f"{subject_id}_{scan_type}_{direction}_{slice_number:03d}.png"

    Image.fromarray(slice_array, mode="L").save(save_path / filename)


def get_scans_in_dir(base_path, pattern):
    return list(base_path.glob(f"*.{pattern}"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Program to slice M4Raw scans"
    )

    parser.add_argument("--data-path", help="The filepath where all the scans are")
    parser.add_argument("--save-path", help="The filepath where all the scans should be saved")
    parser.add_argument("--max-scans", type=int, default=None, help="Max scans to process")


    args = parser.parse_args()

    path = Path(args.data_path)

    if(not path.exists() or not path.is_dir()):
        print("Error: path does not exist or is not a directory")
        exit()

    scans = get_scans_in_dir(path, "zip")

    processed_counter = 0

    print(f"Processing {len(scans)} scans")


    for scan_path in scans:

        if args.max_scans is not None and processed_counter >= args.max_scans:
            break

        # Get scan name
        parts = scan_path.stem.split("_")
        subject_id = parts[0]

        print(f"Processing scan {processed_counter + 1}:           {subject_id}")


        # Load scan
        scan_data, tmp_path = load_scan(scan_path, subject_id)

        # Slice scan
        slices = slice_scan(scan_data)

        # Save slices
        for slice_tuple in slices:
            save_slice(slice_tuple, args.save_path, subject_id, "T1w")

        # Delete temp file
        os.remove(tmp_path)

        processed_counter += 1

        # if processed_counter % 10 == 0:
        #     print(f"Processed {processed_counter} scans")


    print(f"Finished processing {processed_counter} scans")




