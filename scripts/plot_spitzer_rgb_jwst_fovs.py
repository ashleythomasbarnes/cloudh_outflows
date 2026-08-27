#!/usr/bin/env python3
"""Plot the Spitzer RGB image with MIRI and NIRCam FOV contours."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.image import imread
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import binary_fill_holes, map_coordinates


HERE = Path(__file__).resolve().parents[1]
SPITZER_JPEG = HERE / "data" / "cutout-IPAC_P_GLIMPSE360.jpeg"
SPITZER_FITS = HERE / "data" / "cutout-IPAC_P_GLIMPSE360.fits"
JWST_DIR = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Galactic/JWST/cloud_h/"
    "jwst/fits/nanfilled/reprojected"
)
MIRI_FITS = JWST_DIR / "jw07230-o002_t003_miri_f770w_i2d_nanfilled_reprojected.fits"
NIRCAM_FITS = (
    JWST_DIR / "jw07230-o004_t003_nircam_clear-f356w_i2d_nanfilled_reprojected.fits"
)
OUTPUT_FILE = HERE / "plots" / "spitzer_rgb_jwst_fovs.pdf"


# The JPEG uses top-to-bottom rows; FITS/WCS y coordinates start at the bottom.
Image.MAX_IMAGE_PIXELS = None
spitzer_rgb = np.flipud(imread(SPITZER_JPEG)[..., :3])

with fits.open(SPITZER_FITS, memmap=True) as hdul:
    spitzer_shape = hdul[0].data.shape[-2:]
    spitzer_wcs = WCS(hdul[0].header).celestial

if spitzer_rgb.shape[:2] != spitzer_shape:
    raise ValueError(
        f"Spitzer JPEG shape {spitzer_rgb.shape[:2]} does not match FITS shape {spitzer_shape}."
    )

with fits.open(MIRI_FITS, memmap=True) as hdul:
    miri_wcs = WCS(hdul[1].header).celestial
with fits.open(NIRCAM_FITS, memmap=True) as hdul:
    nircam_wcs = WCS(hdul[1].header).celestial


# Convert every Spitzer pixel to sky coordinates, then sample the two JWST
# finite-data masks there. These are the actual FOVs, not rectangular estimates.
height, width = spitzer_shape
spitzer_y, spitzer_x = np.mgrid[0:height, 0:width]
ra, dec = spitzer_wcs.pixel_to_world_values(spitzer_x, spitzer_y)

miri_x, miri_y = miri_wcs.world_to_pixel_values(ra, dec)
with fits.open(MIRI_FITS, memmap=True) as hdul:
    miri_sampled = map_coordinates(
        hdul[1].data,
        [miri_y, miri_x],
        order=0,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
miri_fov = binary_fill_holes(np.isfinite(miri_sampled))

nircam_x, nircam_y = nircam_wcs.world_to_pixel_values(ra, dec)
with fits.open(NIRCAM_FITS, memmap=True) as hdul:
    nircam_sampled = map_coordinates(
        hdul[1].data,
        [nircam_y, nircam_x],
        order=0,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
nircam_fov = binary_fill_holes(np.isfinite(nircam_sampled))


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(1, 1, 1, projection=spitzer_wcs)
ax.imshow(spitzer_rgb, origin="lower")
ax.contour(
    miri_fov.astype(float),
    levels=[0.5],
    colors="white",
    linewidths=1.2,
    linestyles=[(0, (7, 3))],
    origin="lower",
)
ax.contour(
    nircam_fov.astype(float),
    levels=[0.5],
    colors="white",
    linewidths=1.2,
    linestyles=[(0, (2, 2))],
    origin="lower",
)
ax.coords.grid(color="white", alpha=0.45, linestyle=":")
ax.coords[0].set_axislabel("Right Ascension (ICRS)")
ax.coords[1].set_axislabel("Declination (ICRS)")
ax.coords[0].set_major_formatter("hh:mm:ss")
ax.coords[1].set_major_formatter("dd:mm:ss")
ax.coords[0].set_ticklabel(rotation=90, ha="left", va="center")

ax.text(
    0.02,
    0.1,
    "Spitzer 8 $\\mu$m + 5.8 $\\mu$m + 4.5 $\\mu$m",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    color="white",
    bbox=dict(
        boxstyle="round,pad=0.28",
        facecolor="black",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.72,
    ),
)
ax.legend(
    handles=[
        Line2D(
            [0], [0], color="white", linewidth=1.2, linestyle=(0, (7, 3)), label="MIRI FOV"
        ),
        Line2D(
            [0], [0], color="white", linewidth=1.2, linestyle=(0, (2, 2)), label="NIRCam FOV"
        ),
    ],
    loc="lower right",
    facecolor="black",
    edgecolor="white",
    framealpha=0.72,
    labelcolor="white",
    fontsize=9,
)

fig.tight_layout()
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
print(f"Saved {OUTPUT_FILE}")
