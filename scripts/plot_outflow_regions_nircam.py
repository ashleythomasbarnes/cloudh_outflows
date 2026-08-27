#!/usr/bin/env python3
"""Plot the DS9 outflow boxes as rotated, NIRCam-only cutouts."""

from pathlib import Path
import re

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.image import imread
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates


HERE = Path(__file__).resolve().parents[1]
REGION_FILE = HERE / "analysis" / "outflow_forplotting.reg"
JWST_DIR = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Galactic/JWST/cloud_h/"
    "jwst/fits/nanfilled/reprojected"
)
NIRCAM_JPEG = JWST_DIR / "nircam_photoshop.jpeg"
NIRCAM_FITS = (
    JWST_DIR / "jw07230-o004_t003_nircam_clear-f356w_i2d_nanfilled_reprojected.fits"
)
H2_FITS = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Galactic/JWST/cloud_h/"
    "jwst/fits/nanfilled/H2_F470N_minus_F480M_contsub_nanfilled.fits"
)
OUTPUT_DIR = HERE / "plots"
PADDING = 1.0  # Keep this at 1.0 to plot exactly the DS9 box size.
FIGURE_WIDTH_INCHES = 3.4  # Single-column width. Adjust if needed.
H2_LEVELS = [0.1, 0.3]
H2_SMOOTHING_SIGMA_PIXELS = 10


# JPEG row 0 is at the top, whereas FITS/WCS y=0 is at the bottom.
Image.MAX_IMAGE_PIXELS = None
nircam_rgb = np.flipud(imread(NIRCAM_JPEG)[..., :3])

with fits.open(NIRCAM_FITS, memmap=True) as hdul:
    nircam_shape = hdul[1].data.shape
    nircam_wcs = WCS(hdul[1].header).celestial

with fits.open(H2_FITS, memmap=True) as hdul:
    h2_wcs = WCS(hdul[0].header).celestial

if nircam_rgb.shape[:2] != nircam_shape:
    raise ValueError(
        f"NIRCam JPEG shape {nircam_rgb.shape[:2]} does not match FITS shape {nircam_shape}."
    )

nircam_pixel_scale = (
    np.mean(proj_plane_pixel_scales(nircam_wcs)) * u.deg
).to_value(u.arcsec)

region_lines = [
    line.strip() for line in REGION_FILE.read_text().splitlines() if line.startswith("box(")
]

for region_number, line in enumerate(region_lines, start=1):
    # The region file contains only FK5 boxes: RA, Dec, width, height, angle.
    values = re.search(r"box\(([^)]+)\)", line).group(1).split(",")
    ra_deg = float(values[0])
    dec_deg = float(values[1])
    width_arcsec = float(values[2].replace('"', ""))
    height_arcsec = float(values[3].replace('"', ""))
    box_angle_deg = float(values[4])

    # A DS9 angle is counter-clockwise from the WCS x axis (east). The height
    # axis is therefore at the quoted angle from north, while the width axis is
    # 90 degrees east of it. Choose the longer axis as the new horizontal axis.
    if width_arcsec >= height_arcsec:
        long_axis_arcsec = width_arcsec
        short_axis_arcsec = height_arcsec
        long_axis_pa_deg = box_angle_deg + 90
    else:
        long_axis_arcsec = height_arcsec
        short_axis_arcsec = width_arcsec
        long_axis_pa_deg = box_angle_deg

    output_width = int(np.ceil(PADDING * long_axis_arcsec / nircam_pixel_scale))
    output_height = int(np.ceil(PADDING * short_axis_arcsec / nircam_pixel_scale))
    region_center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="fk5")

    # Make an output WCS whose positive x axis follows the long box axis.
    wcs_rotation_rad = np.deg2rad(long_axis_pa_deg + 90)
    rotation_matrix = np.array(
        [
            [np.cos(wcs_rotation_rad), -np.sin(wcs_rotation_rad)],
            [np.sin(wcs_rotation_rad), np.cos(wcs_rotation_rad)],
        ]
    )
    cutout_wcs = WCS(naxis=2)
    cutout_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    cutout_wcs.wcs.cunit = ["deg", "deg"]
    cutout_wcs.wcs.crval = [region_center.ra.deg, region_center.dec.deg]
    cutout_wcs.wcs.crpix = [(output_width + 1) / 2, (output_height + 1) / 2]
    cutout_wcs.wcs.cdelt = [nircam_pixel_scale / 3600, nircam_pixel_scale / 3600]
    cutout_wcs.wcs.pc = np.array([[-1, 0], [0, 1]]) @ rotation_matrix
    cutout_wcs.wcs.set()

    cutout_y, cutout_x = np.mgrid[0:output_height, 0:output_width]
    ra, dec = cutout_wcs.pixel_to_world_values(cutout_x, cutout_y)
    nircam_x, nircam_y = nircam_wcs.world_to_pixel_values(ra, dec)
    h2_x, h2_y = h2_wcs.world_to_pixel_values(ra, dec)

    cutout_rgb = np.empty((output_height, output_width, 3), dtype=nircam_rgb.dtype)
    for channel in range(3):
        cutout_rgb[..., channel] = map_coordinates(
            nircam_rgb[..., channel],
            [nircam_y, nircam_x],
            order=1,
            mode="constant",
            cval=0,
        )

    # Reproject H2 onto the rotated NIRCam grid, then smooth only this small
    # cutout. The weight map prevents NaNs from spreading through the result.
    with fits.open(H2_FITS, memmap=True) as hdul:
        h2_cutout = map_coordinates(
            hdul[0].data,
            [h2_y, h2_x],
            order=1,
            mode="constant",
            cval=np.nan,
        ).astype(np.float32)
    h2_valid = np.isfinite(h2_cutout)
    h2_smoothed_numerator = gaussian_filter(
        np.where(h2_valid, h2_cutout, 0.0), sigma=H2_SMOOTHING_SIGMA_PIXELS
    )
    h2_smoothed_weight = gaussian_filter(
        h2_valid.astype(np.float32), sigma=H2_SMOOTHING_SIGMA_PIXELS
    )
    h2_smoothed = np.full(h2_cutout.shape, np.nan, dtype=np.float32)
    np.divide(
        h2_smoothed_numerator,
        h2_smoothed_weight,
        out=h2_smoothed,
        where=h2_smoothed_weight > 0,
    )

    figure_height_inches = FIGURE_WIDTH_INCHES * output_height / output_width
    fig = plt.figure(figsize=(FIGURE_WIDTH_INCHES, figure_height_inches))
    ax = fig.add_subplot(1, 1, 1, projection=cutout_wcs)
    ax.set_facecolor("black")
    ax.imshow(cutout_rgb, origin="lower")
    ax.contour(
        h2_smoothed,
        levels=H2_LEVELS,
        colors="cyan",
        linewidths=0.9,
        origin="lower",
    )

    # Keep coordinate ticks and a grid, but omit RA/Dec text and axis labels.
    ax.coords.grid(color="white", alpha=0.35, linestyle=":", linewidth=0.7)
    ax.coords[0].set_ticks(color="white")
    ax.coords[1].set_ticks(color="white")
    ax.coords[0].set_ticklabel_visible(False)
    ax.coords[1].set_ticklabel_visible(False)
    ax.coords[0].set_axislabel("")
    ax.coords[1].set_axislabel("")

    fig.tight_layout(pad=0.05)
    output_file = OUTPUT_DIR / f"nircam_outflow_region_{region_number:02d}.pdf"
    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {output_file}")
