# Step 4: Characterize cutout properties (image depth, PSF, etc) by SExtractor 
# TODO: Ran this part in wsl ubuntu with default.conv, default.param, default.sex, and default.nnw files

import os
import csv
import uuid
import requests
import subprocess
import tempfile
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from scipy.ndimage import shift
from collections import defaultdict
from glob import glob

# === Configuration ===
BANDS = ['g', 'r', 'i']
CUTOUT_FOLDER = "cutouts"
PSF_OUTPUT_FOLDER = "psf_models"
STAR_SIZE = 25
FWHM_BIN_WIDTH = 0.05
MIN_FWHM = 0.5
MAX_FWHM = 1.5
SEX_CONFIG = "default.sex"
SEX_PARAMS = "default.param"
SEX_CLASS_STAR_CUTOFF = 0.9
SEX_DETECT_THRESH = 3.0
PSF_STACK_MIN = 9
PSF_STACK_MAX = 11

os.makedirs(PSF_OUTPUT_FOLDER, exist_ok=True)

# === Query PanSTARRS PSF FWHM from MAST ===
def query_ps1_psf_fwhm(ra, dec, band):
    base_url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/stack.csv"
    major_col = f"{band}psfMajorFWHM"
    minor_col = f"{band}psfMinorFWHM"
    params = {
        "ra": ra,
        "dec": dec,
        "radius": 0.02,  # ~1.2 arcmin
        "nDetections.gt": 0,
        "columns": f"raMean,decMean,{major_col},{minor_col}"
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().split("\n")
        if len(lines) < 2:
            print("No data rows returned from PS1 query.")
            return None

        header = lines[0].split(",")
        try:
            major_idx = header.index(major_col)
            minor_idx = header.index(minor_col)
        except ValueError:
            print(f"Columns {major_col} or {minor_col} not found in PS1 response header.")
            return None

        fwhm_values = []
        for line in lines[1:]:
            row = line.split(",")
            try:
                major_val = float(row[major_idx])
                minor_val = float(row[minor_idx])
                # Filter out invalid values (like -999 or unphysical values)
                if 0 < major_val < 10 and 0 < minor_val < 10:
                    fwhm_values.append((major_val + minor_val) / 2)
            except (ValueError, IndexError):
                continue

        if not fwhm_values:
            print(f"No valid FWHM data found for band '{band}'.")
            return None

        avg_fwhm = sum(fwhm_values) / len(fwhm_values)
        return avg_fwhm
    except Exception as e:
        print(f"PS1 query failed: {e}")
        return None

# === Run SExtractor ===
def run_sextractor(fits_path):
    catalog_path = os.path.join(tempfile.gettempdir(), f"sex_{uuid.uuid4().hex}.cat")
    cmd = [
        "sex", fits_path,
        "-c", SEX_CONFIG,
        "-PARAMETERS_NAME", SEX_PARAMS,
        "-DETECT_THRESH", str(SEX_DETECT_THRESH),
        "-CATALOG_NAME", catalog_path,
        "-CATALOG_TYPE", "ASCII_HEAD"
    ]
    try:
        subprocess.run(cmd, check=True)
        if not os.path.exists(catalog_path):
            print(f"SExtractor did not produce catalog for {fits_path}")
            return None
        data = np.genfromtxt(catalog_path, names=["X_IMAGE", "Y_IMAGE", "FWHM_IMAGE", "CLASS_STAR", "FLUX_AUTO"])
        if data.shape == ():  # single detection
            data = np.array([data])
        os.remove(catalog_path)
        return data
    except subprocess.CalledProcessError as e:
        print(f"SExtractor failed on {fits_path} with error: {e}")
        return None

# === Extract PSF cutout ===
def extract_psf_cutout(hdu_data, x, y, size=STAR_SIZE):
    try:
        cutout = Cutout2D(hdu_data, (x, y), (size, size), mode='partial', fill_value=0)
        data = cutout.data
        y_peak, x_peak = np.unravel_index(np.argmax(data), data.shape)
        shift_y, shift_x = (size // 2 - y_peak, size // 2 - x_peak)
        centered = shift(data, shift=(shift_y, shift_x), order=1)
        norm = np.sum(centered)
        return centered / norm if norm != 0 else centered
    except Exception as e:
        print(f"Cutout failed at position ({x}, {y}): {e}")
        return None

# === Estimate image depth ===
def estimate_image_depth(data, header):
    flattened = data.flatten()
    sigma = np.std(flattened)
    zp_keys = [key for key in header.keys() if key.startswith("ZPT_")]
    zp_values = [header[key] for key in zp_keys if isinstance(header[key], (float, int))]
    zp_avg = np.mean(zp_values) if zp_values else 25.0
    depth = zp_avg - 2.5 * np.log10(5 * sigma) if sigma > 0 else None
    return sigma, depth, zp_avg

# === Main processor ===
# === Main processor ===
def process_fits_file(fits_path, band):
    basename = os.path.splitext(os.path.basename(fits_path))[0]
    output_dir = os.path.join(PSF_OUTPUT_FOLDER, band)
    csv_file = os.path.join(output_dir, "psf_summary.csv")
    os.makedirs(output_dir, exist_ok=True)
    cutouts_by_bin = defaultdict(list)

    try:
        sex_result = run_sextractor(fits_path)
        if sex_result is None or len(sex_result) == 0:
            print(f"No detections for {fits_path}")
            return

        with fits.open(fits_path) as hdu:
            data = hdu[0].data
            header = hdu[0].header

        sigma, depth, zp_avg = estimate_image_depth(data, header)

        ra = header.get("RA") or header.get("CRVAL1")
        dec = header.get("DEC") or header.get("CRVAL2")

        ps1_psf_fwhm = query_ps1_psf_fwhm(ra, dec, band) if ra and dec else None

        print(f"\n=== {fits_path} ===")
        print(f"Estimated image depth (proxy): {depth} mag | RMS: {sigma} | Avg ZP: {zp_avg} | PS1 Seeing FWHM: {ps1_psf_fwhm if ps1_psf_fwhm is not None else 'N/A'}")

        # === NEW: skip if FWHM or depth missing ===
        if ps1_psf_fwhm is None:
            print(f"Skipping CSV write for {fits_path} — no PS1 seeing FWHM available.")
            return

        if depth is None:
            print(f"Skipping CSV write for {fits_path} — image depth could not be estimated.")
            return

        specobjid = header.get("SPECOBJID", basename)

        star_count = 0
        for row in sex_result:
            if row['CLASS_STAR'] < SEX_CLASS_STAR_CUTOFF:
                continue
            fwhm = row['FWHM_IMAGE']
            if not (MIN_FWHM <= fwhm <= MAX_FWHM):
                continue
            bin_index = round(fwhm / FWHM_BIN_WIDTH) * FWHM_BIN_WIDTH
            cutout = extract_psf_cutout(data, row['X_IMAGE'], row['Y_IMAGE'])
            if cutout is not None:
                cutouts_by_bin[bin_index].append(cutout)
                star_count += 1

        print(f"Stars passing quality cuts: {star_count}")

        for fwhm_bin, cutouts in cutouts_by_bin.items():
            if PSF_STACK_MIN <= len(cutouts) <= PSF_STACK_MAX:
                stack = np.median(np.array(cutouts), axis=0)
                out_path = os.path.join(output_dir, f"psf_model_{fwhm_bin:.2f}.fits")
                fits.writeto(out_path, stack, overwrite=True)
                print(f"Saved PSF model to {out_path} with {len(cutouts)} stars")
            else:
                print(f"Skipping bin {fwhm_bin:.2f} — only {len(cutouts)} stars (need {PSF_STACK_MIN}–{PSF_STACK_MAX})")

        write_header = not os.path.exists(csv_file)
        with open(csv_file, "a", newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["specobjid", "image_depth", "ps1_seeing_fwhm"])
            writer.writerow([specobjid, depth, ps1_psf_fwhm])

    except Exception as e:
        print(f"Failed to process {fits_path}: {e}")

# === Batch processor ===
def build_psf_library_serial():
    for band in BANDS:
        fits_files = sorted(glob(os.path.join(CUTOUT_FOLDER, band, "*.fits")))
        print(f"\nFound {len(fits_files)} files in '{band}' band.")
        for fits_path in fits_files:
            process_fits_file(fits_path, band)

# Process all fits files in cutouts/g, cutouts/r, cutouts/i automatically
for band in ['g', 'r', 'i']:
    fits_files = sorted(glob(os.path.join('cutouts', band, '*.fits')))
    print(f"\nFound {len(fits_files)} files in '{band}' band.")
    for fits_file in fits_files:
        print(f"\nProcessing {fits_file} in band '{band}'")
        process_fits_file(fits_file, band)




