#!/usr/bin/env python3
"""Fill small, internal NaN islands in JWST science images.

NaN regions that touch an image edge are left unchanged, so the script never
extrapolates beyond the observed footprint.  The default 5 x 5 pixel Gaussian
kernel has a 1-pixel standard deviation, appropriate for the small gaps in
these maps rather than broad spatial interpolation.
"""

from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from scipy.ndimage import find_objects, label


DEFAULT_DATA_DIR = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Galactic/JWST/"
    "cloud_h/jwst/fits"
)


def internal_nan_components(data):
    """Return enclosed NaN-component masks, excluding the image footprint."""
    nan_mask = np.isnan(data)
    labels, count = label(nan_mask, structure=np.ones((3, 3), dtype=int))
    ny, nx = data.shape
    components = []

    for component, component_slice in enumerate(find_objects(labels), start=1):
        if component_slice is None:
            continue
        y_slice, x_slice = component_slice
        component_mask = labels[component_slice] == component
        touches_edge = (
            y_slice.start == 0
            or y_slice.stop == ny
            or x_slice.start == 0
            or x_slice.stop == nx
        )
        if not touches_edge:
            components.append((int(component_mask.sum()), component_slice, component_mask))

    return components


def small_internal_nan_islands(data, max_pixels):
    """Return small enclosed NaN islands suitable for local interpolation."""
    fill_mask = np.zeros(data.shape, dtype=bool)
    for size, component_slice, component_mask in internal_nan_components(data):
        if size <= max_pixels:
            fill_mask[component_slice] |= component_mask

    return fill_mask


def fill_small_internal_nans(data, kernel, max_pixels):
    """Interpolate eligible NaNs with Astropy, leaving every other NaN intact."""
    candidate_mask = small_internal_nan_islands(data, max_pixels)
    if not candidate_mask.any():
        return np.array(data, copy=True), candidate_mask

    # Large NaN regions outside the footprint are intentionally retained.  They
    # trigger Astropy's broad-NaN warning but cannot affect enclosed islands.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        interpolated = interpolate_replace_nans(data, kernel)
    fill_mask = candidate_mask & np.isfinite(interpolated)
    result = np.array(data, copy=True)
    result[fill_mask] = interpolated[fill_mask]
    return result, fill_mask


def fill_large_internal_nans(data, kernel, min_pixels):
    """Fill every large enclosed NaN island with repeated local interpolation."""
    result = np.array(data, copy=True)
    n_filled = 0
    n_islands = 0
    radius = kernel.array.shape[0] // 2
    padding = kernel.array.shape[0]

    for size, component_slice, component_mask in internal_nan_components(data):
        if size <= min_pixels:
            continue
        n_islands += 1
        y_slice, x_slice = component_slice
        y0 = max(0, y_slice.start - padding)
        y1 = min(data.shape[0], y_slice.stop + padding)
        x0 = max(0, x_slice.start - padding)
        x1 = min(data.shape[1], x_slice.stop + padding)
        local = np.array(result[y0:y1, x0:x1], copy=True)
        local_mask = np.zeros(local.shape, dtype=bool)
        local_mask[
            y_slice.start - y0:y_slice.stop - y0,
            x_slice.start - x0:x_slice.stop - x0,
        ] = component_mask

        # A convolution fills roughly one kernel radius inward per pass. Work
        # in each small cutout so the broad pass remains practical for mosaics.
        passes = int(np.ceil(max(component_mask.shape) / (2 * radius)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyUserWarning)
            for _ in range(max(1, passes)):
                interpolated = interpolate_replace_nans(local, kernel)
                local[local_mask] = interpolated[local_mask]

        fill_mask = local_mask & np.isfinite(local)
        result_view = result[y0:y1, x0:x1]
        result_view[fill_mask] = local[fill_mask]
        n_filled += int(fill_mask.sum())

    return result, n_filled, n_islands


def process_file(filename, outdir, small_kernel, large_kernel, max_pixels, dry_run):
    with fits.open(filename, memmap=False) as hdul:
        if "SCI" not in hdul:
            print(f"Skipping {filename.name}: no SCI extension")
            return

        science = hdul["SCI"].data
        if science.ndim != 2:
            print(f"Skipping {filename.name}: SCI is {science.ndim}-dimensional")
            return

        print(f"Processing {filename.name} ...", flush=True)
        filled, fill_mask = fill_small_internal_nans(science, small_kernel, max_pixels)
        filled, n_large_filled, n_large_islands = fill_large_internal_nans(
            filled, large_kernel, max_pixels
        )
        n_filled = int(fill_mask.sum()) + n_large_filled
        n_nans = int(np.isnan(science).sum())
        action = "would fill" if dry_run else "filled"
        print(
            f"{filename.name}: {action} {n_filled} of {n_nans} NaNs "
            f"({int(fill_mask.sum())} small, {n_large_filled} in "
            f"{n_large_islands} large islands)"
        )
        if dry_run:
            return

        header = hdul["SCI"].header.copy()
        header["NANFILL"] = (True, "All enclosed NaN islands interpolated")
        header["NFKSTD"] = (small_kernel.model.x_stddev.value, "Small NaN-fill kernel stddev (pix)")
        header["NFKXSZ"] = (small_kernel.array.shape[1], "Small NaN-fill kernel width (pix)")
        header["NLKSTD"] = (large_kernel.model.x_stddev.value, "Large NaN-fill kernel stddev (pix)")
        header["NLKXSZ"] = (large_kernel.array.shape[1], "Large NaN-fill kernel width (pix)")
        header["NFMAXPX"] = (max_pixels, "Small-island threshold (pixels)")
        header.add_history("Only enclosed NaN islands were filled; edge-connected NaNs retained.")
        output = outdir / f"{filename.stem}_nanfilled.fits"
        fits.PrimaryHDU(
            data=filled.astype(science.dtype, copy=False), header=header
        ).writeto(output, overwrite=True)
        print(f"  wrote {output}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outdir", type=Path, help="Default: DATA_DIR/nanfilled")
    parser.add_argument("--kernel-stddev", type=float, default=1.0)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--large-kernel-stddev", type=float, default=8.0)
    parser.add_argument("--large-kernel-size", type=int, default=21)
    parser.add_argument("--max-pixels", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if any(size < 3 or size % 2 == 0 for size in (args.kernel_size, args.large_kernel_size)):
        parser.error("Kernel sizes must be odd integers of at least 3")
    if args.kernel_stddev <= 0 or args.large_kernel_stddev <= 0 or args.max_pixels < 1:
        parser.error("Kernel standard deviations and --max-pixels must be positive")

    files = sorted(args.data_dir.glob("*.fits"))
    if not files:
        parser.error(f"No .fits files found in {args.data_dir}")
    outdir = args.outdir or args.data_dir / "nanfilled"
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    small_kernel = Gaussian2DKernel(args.kernel_stddev, x_size=args.kernel_size, y_size=args.kernel_size)
    large_kernel = Gaussian2DKernel(
        args.large_kernel_stddev,
        x_size=args.large_kernel_size,
        y_size=args.large_kernel_size,
    )
    for filename in files:
        process_file(filename, outdir, small_kernel, large_kernel, args.max_pixels, args.dry_run)


if __name__ == "__main__":
    main()
