# Step 4: Characterize cutout properties of lenses (image depth, PSF, centroid, axis ratio, position angle) by SExtractor 

import os
import csv
import uuid
import requests
import subprocess
import tempfile
from glob import glob
from collections import defaultdict
from multiprocessing import Pool, Manager
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from scipy.ndimage import shift
from lmfit import Model, Parameters
from astropy.modeling import models

# === Config ===
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


# === Shared PS1 FWHM cache ===
ps1_cache = Manager().dict()  # shared among processes


def query_ps1_psf_fwhm(ra, dec, band):
    # round RA/Dec to 3 decimal to reduce identical queries
    key = (round(ra, 3), round(dec, 3), band)
    if key in ps1_cache:
        return ps1_cache[key]

    base_url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/stack.csv"
    major_col = f"{band}psfMajorFWHM"
    minor_col = f"{band}psfMinorFWHM"
    params = {
        "ra": ra,
        "dec": dec,
        "radius": 0.02,
        "nDetections.gt": 0,
        "columns": f"raMean,decMean,{major_col},{minor_col}"
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        lines = response.text.strip().split("\n")
        if len(lines) < 2:
            print("No data rows returned from PS1 query.")
            ps1_cache[key] = None
            return None

        header = lines[0].split(",")
        major_idx = header.index(major_col)
        minor_idx = header.index(minor_col)

        fwhm_values = []
        for line in lines[1:]:
            row = line.split(",")
            try:
                major_val = float(row[major_idx])
                minor_val = float(row[minor_idx])
                if 0 < major_val < 10 and 0 < minor_val < 10:
                    fwhm_values.append((major_val + minor_val) / 2)
            except (ValueError, IndexError):
                continue

        if not fwhm_values:
            print(f"No valid FWHM data found for band '{band}'.")
            ps1_cache[key] = None
            return None

        avg_fwhm = sum(fwhm_values) / len(fwhm_values)
        ps1_cache[key] = avg_fwhm
        return avg_fwhm
    except Exception as e:
        print(f"PS1 query failed: {e}")
        ps1_cache[key] = None
        return None


# === SExtractor ===
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

        if os.path.getsize(catalog_path) == 0:
            print(f"SExtractor produced empty catalog for {fits_path}")
            return None

        data = np.genfromtxt(catalog_path, names=["X_IMAGE", "Y_IMAGE", "FWHM_IMAGE", "CLASS_STAR", "FLUX_AUTO"])
        if data.shape == ():  # single detection
            data = np.array([data])
        os.remove(catalog_path)
        return data

    except subprocess.CalledProcessError as e:
        print(f"SExtractor failed on {fits_path} with error: {e}")
        return None


# === PSF cutout ===
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


# === Image depth ===
def estimate_image_depth(data, header):
    flattened = data.flatten()
    sigma = np.std(flattened)
    zp_keys = [key for key in header.keys() if key.startswith("ZPT_")]
    zp_values = [header[key] for key in zp_keys if isinstance(header[key], (float, int))]
    zp_avg = np.mean(zp_values) if zp_values else 25.0
    depth = zp_avg - 2.5 * np.log10(5 * sigma) if sigma > 0 else None
    return sigma, depth, zp_avg


# === Main processing ===
def process_fits_file(args):
    fits_path, band = args  # unpack for pool.map

    basename = os.path.splitext(os.path.basename(fits_path))[0]
    output_dir = os.path.join(PSF_OUTPUT_FOLDER, band)
    os.makedirs(output_dir, exist_ok=True)
    cutouts_by_bin = defaultdict(list)
    csv_file = os.path.join(output_dir, "psf_summary.csv")

    specobjid = basename

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
        print(f"Estimated image depth: {depth} mag | RMS: {sigma} | Avg ZP: {zp_avg} | PS1 Seeing FWHM: {ps1_psf_fwhm}")

        if ps1_psf_fwhm is None or depth is None:
            print(f"Skipping CSV update for {fits_path} — missing PS1 seeing FWHM or depth.")
            return

        ny, nx = data.shape
        Y, X = np.mgrid[:ny, :nx]

        def sersic2d_lmfit(X, Y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
            model = models.Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n,
                                    x_0=x_0, y_0=y_0, ellip=ellip, theta=theta)
            return model(X, Y).ravel()

        sersic_model = Model(sersic2d_lmfit, independent_vars=['X', 'Y'])

        params = Parameters()
        params.add('amplitude', value=data.max(), min=0)
        params.add('r_eff', value=30, min=1)
        params.add('n', value=4.0, min=0.5, max=8)
        params.add('x_0', value=nx/2, min=0, max=nx)
        params.add('y_0', value=ny/2, min=0, max=ny)
        params.add('ellip', value=0.2, min=0, max=0.99)
        params.add('theta', value=np.random.uniform(0, np.pi), min=0, max=np.pi)

        result = sersic_model.fit(data.ravel(), params, X=X, Y=Y)

        centroid_x = result.params['x_0'].value
        centroid_y = result.params['y_0'].value
        axis_ratio_q = 1 - result.params['ellip'].value
        position_angle_deg = np.degrees(result.params['theta'].value)

        print(f"Lens (LMFIT Sersic): Centroid ({centroid_x}, {centroid_y}), "
              f"q={axis_ratio_q}, PA={position_angle_deg} deg")

        # PSF stars
        star_count = 0
        for row_star in sex_result:
            if row_star['CLASS_STAR'] < SEX_CLASS_STAR_CUTOFF:
                continue
            fwhm = row_star['FWHM_IMAGE']
            if not (MIN_FWHM <= fwhm <= MAX_FWHM):
                continue
            bin_index = round(fwhm / FWHM_BIN_WIDTH) * FWHM_BIN_WIDTH
            cutout = extract_psf_cutout(data, row_star['X_IMAGE'], row_star['Y_IMAGE'])
            if cutout is not None:
                cutouts_by_bin[bin_index].append(cutout)
                star_count += 1

        print(f"Stars for PSF: {star_count}")

        for fwhm_bin, cutouts in cutouts_by_bin.items():
            if PSF_STACK_MIN <= len(cutouts) <= PSF_STACK_MAX:
                stack = np.median(np.array(cutouts), axis=0)
                out_path = os.path.join(output_dir, f"psf_model_{fwhm_bin}.fits")
                fits.writeto(out_path, stack, overwrite=True)
                print(f"Saved PSF: {out_path} ({len(cutouts)} stars)")

        # Append CSV
        write_header = not os.path.exists(csv_file)
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(["specobjid", "image_depth", "ps1_seeing_fwhm",
                                 "centroid_x", "centroid_y", "axis_ratio_q", "position_angle_deg"])
            writer.writerow([specobjid, depth, ps1_psf_fwhm,
                             centroid_x, centroid_y, axis_ratio_q, position_angle_deg])

        print(f"Appended CSV row for {specobjid}")

    except Exception as e:
        print(f"Failed on {fits_path}: {e}")


# === Batch processor with Pool ===
def build_psf_library_parallel():
    tasks = []
    for band in BANDS:
        fits_files = sorted(glob(os.path.join(CUTOUT_FOLDER, band, "*.fits")))
        for fits_path in fits_files:
            tasks.append((fits_path, band))

    with Pool(processes=8) as pool:  # adjust for your CPU cores
        pool.map(process_fits_file, tasks)


if __name__ == "__main__":
    build_psf_library_parallel()





