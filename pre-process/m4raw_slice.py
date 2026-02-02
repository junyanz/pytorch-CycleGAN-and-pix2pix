# Slices the 0.3T scans from the m4raw dataset

# For each scan type, extracts max slices from axial, coronal and sagittal directions

# Saves files to specified directory in sub directory for direction file name format: <subjectID>_<scanType>_<sliceDirection>_<sliceNumber>

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import h5py






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


def load_scan(path):

    # Load h5 file
    scan_file = h5py.File(path, "r")

    # Load data into numpy array
    volume = np.array(scan_file['reconstruction_rss'])

    slice_axis = np.argmin(volume.shape)
    
    # Move slice axis to last position
    axes = list(range(3))
    axes.remove(slice_axis)
    axes.append(slice_axis)
    volume = np.transpose(volume, axes)

    return volume



def save_slice(slice_tuple, session_id, save_path, scan_type):
    slice_array, direction, slice_number = slice_tuple

    # Convert to float32
    slice_array = slice_array.astype(np.float32)

    # Normalize to 0-1
    slice_array -= slice_array.min()
    max_val = slice_array.max()
    if max_val > 0:
        slice_array /= max_val

    # Scale to [-1, 1]
    slice_array = slice_array * 2.0 - 1.0

    # For padding/resizing, map back to 0-256
    pil_array = ((slice_array + 1.0) / 2.0 * 256).astype(np.uint8)
    img = Image.fromarray(pil_array, mode="L")  # single-channel

    # Pad to square
    w, h = img.size
    max_dim = max(w, h)
    padded = Image.new("L", (max_dim, max_dim), color=0)  # black padding
    padded.paste(img, ((max_dim - w)//2, (max_dim - h)//2))

    # Resize to target size
    img_resized = padded.resize((256, 256), resample=Image.BILINEAR)

    # Save
    save_path = Path(save_path) / direction
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f"{session_id}_{scan_type}_{direction}_{slice_number:03d}.png"
    img_resized.save(save_path / filename)

def get_scans_in_dir(base_path, pattern):
    return list(base_path.glob(f"*{pattern}"))



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

    scans = get_scans_in_dir(path, "_T101.h5") # Get the first-session T1 files

    processed_counter = 0

    print(f"Processing {len(scans)} scans")


    for scan_path in scans:

        if args.max_scans is not None and processed_counter >= args.max_scans:
            break
    
        # Get scan name
        parts = scan_path.stem.split("_")
        session_id = parts[0]

        print(f"Processing scan {processed_counter + 1}")


        # Load scan
        scan_data = load_scan(scan_path)

        # Slice scan
        slices = slice_scan(scan_data)

        # Save slices
        for slice_tuple in slices:
            save_slice(slice_tuple, session_id, args.save_path, "T1w")

        processed_counter += 1

        # if processed_counter % 10 == 0:
        #     print(f"Processed {processed_counter} scans")


    print(f"Finished processing {processed_counter} scans")




