#!/usr/bin/env python3
"""Make an F470N-minus-F480M JWST H2 continuum-subtracted map."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats

try:
    from photutils.detection import DAOStarFinder
except ImportError as error:
    raise ImportError(
        "This script requires photutils. Install it with "
        "'python -m pip install photutils'."
    ) from error

from scipy.ndimage import binary_dilation, find_objects, label
from scipy.spatial import cKDTree

try:
    from reproject import reproject_interp
except ImportError as error:
    raise ImportError(
        "This script requires reproject. Install it with "
        "'python -m pip install reproject'."
    ) from error


DEFAULT_DATA_DIR = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Galactic/JWST/"
    "cloud_h/jwst/fits"
)
LINE_NAME = "jw07230-o004_t003_nircam_f444w-f470n_i2d.fits"
CONT_NAME = "jw07230-o004_t003_nircam_clear-f480m_i2d.fits"
OUTPUT_NAME = "H2_F470N_minus_F480M_contsub.fits"
TILE_SIZE = 2048


def load_image(filename):
    """Load the science image, its uncertainty, and FITS headers."""
    with fits.open(filename, memmap=True) as hdul:
        science = np.asarray(hdul["SCI"].data, dtype=np.float32)
        error = np.asarray(hdul["ERR"].data, dtype=np.float32)
        image_header = hdul["SCI"].header.copy()
        primary_header = hdul[0].header.copy()
    return science, error, image_header, primary_header


def reproject_image_and_error(filename, target_header):
    """Reproject a continuum image and its variance to the target WCS."""
    with fits.open(filename, memmap=True) as hdul:
        source_header = hdul["SCI"].header
        continuum, footprint = reproject_interp(
            (hdul["SCI"].data, source_header), target_header
        )
        variance = np.square(
            np.asarray(hdul["ERR"].data, dtype=np.float32), dtype=np.float32
        )
        reprojected_variance, variance_footprint = reproject_interp(
            (variance, source_header), target_header
        )

    footprint = np.minimum(footprint, variance_footprint)
    continuum = np.asarray(continuum, dtype=np.float32)
    continuum_error = np.sqrt(
        np.asarray(reprojected_variance, dtype=np.float32), dtype=np.float32
    )
    invalid = footprint <= 0
    continuum[invalid] = np.nan
    continuum_error[invalid] = np.nan
    return continuum, continuum_error, np.asarray(footprint, dtype=np.float32)


def estimate_continuum_scale(line, line_error, continuum, continuum_error, min_snr):
    """Measure a clipped global F470N/F480M ratio from well-measured pixels."""
    # Sampling every fourth pixel keeps the ratio calculation small while still
    # using millions of independent locations in the full JWST mosaics.
    line = line[::4, ::4]
    line_error = line_error[::4, ::4]
    continuum = continuum[::4, ::4]
    continuum_error = continuum_error[::4, ::4]

    valid = (
        np.isfinite(line)
        & np.isfinite(line_error)
        & np.isfinite(continuum)
        & np.isfinite(continuum_error)
        & (line > 0)
        & (continuum > 0)
        & (line_error > 0)
        & (continuum_error > 0)
        & (line / line_error >= min_snr)
        & (continuum / continuum_error >= min_snr)
    )
    ratios = line[valid] / continuum[valid]
    if ratios.size == 0:
        raise ValueError("No pixels satisfy the continuum-scale selection")

    clipped = sigma_clip(ratios, sigma=3.0, maxiters=5, masked=False)
    clipped = clipped[np.isfinite(clipped)]
    if clipped.size == 0:
        raise ValueError("No valid pixels remain after continuum-scale clipping")

    scale = float(np.median(clipped))
    scatter = float(1.4826 * np.median(np.abs(clipped - scale)))
    return scale, scatter, len(clipped)


def tile_bounds(shape, overlap):
    """Yield core and padded tile slices for large-image source finding."""
    ny, nx = shape
    for y0 in range(0, ny, TILE_SIZE):
        for x0 in range(0, nx, TILE_SIZE):
            y1 = min(y0 + TILE_SIZE, ny)
            x1 = min(x0 + TILE_SIZE, nx)
            py0 = max(0, y0 - overlap)
            py1 = min(ny, y1 + overlap)
            px0 = max(0, x0 - overlap)
            px1 = min(nx, x1 + overlap)
            yield (y0, y1, x0, x1), (py0, py1, px0, px1)


def find_point_sources(data, fwhm, threshold_sigma, signs=(1,)):
    """Find point-like positive or negative peaks without loading one huge tile."""
    overlap = int(np.ceil(4.0 * fwhm))
    positions = []
    for (y0, y1, x0, x1), (py0, py1, px0, px1) in tile_bounds(
        data.shape, overlap
    ):
        tile = data[py0:py1, px0:px1]
        valid = np.isfinite(tile)
        if valid.sum() < 100:
            continue
        for sign in signs:
            signed_tile = sign * tile
            _, median, std = sigma_clipped_stats(
                signed_tile[valid], sigma=3.0, maxiters=5
            )
            if not np.isfinite(std) or std <= 0:
                continue
            finder = DAOStarFinder(
                fwhm=fwhm, threshold=threshold_sigma * std, exclude_border=True
            )
            sources = finder(signed_tile - median, mask=~valid)
            if sources is None:
                continue
            x = np.asarray(sources["xcentroid"], dtype=float) + px0
            y = np.asarray(sources["ycentroid"], dtype=float) + py0
            in_core = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            positions.extend(zip(x[in_core], y[in_core], strict=True))
    if not positions:
        return np.empty((0, 2), dtype=float)
    return np.asarray(positions, dtype=float)


def match_residual_sources(residual_positions, continuum_positions, max_distance):
    """Keep residual sources that coincide with a continuum point source."""
    if len(residual_positions) == 0 or len(continuum_positions) == 0:
        return np.empty((0, 2), dtype=float)
    distances, _ = cKDTree(continuum_positions).query(residual_positions)
    return residual_positions[distances <= max_distance]


def local_stats(data, mask):
    """Return a sigma-clipped median and scatter for finite unmasked pixels."""
    values = data[np.isfinite(data) & ~mask]
    if values.size < 100:
        return np.nan, np.nan
    _, median, std = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    return float(median), float(std)


def make_source_mask(data, positions, fwhm):
    """Mask stellar cores plus significant nearby residual PSF structure."""
    mask = np.zeros(data.shape, dtype=bool)
    core_radius = 1.5 * fwhm
    max_radius = 4.0 * fwhm
    analysis_radius = 6.0 * fwhm
    ny, nx = data.shape
    for xcenter, ycenter in positions:
        x0 = max(0, int(np.floor(xcenter - analysis_radius)))
        x1 = min(nx, int(np.ceil(xcenter + analysis_radius)) + 1)
        y0 = max(0, int(np.floor(ycenter - analysis_radius)))
        y1 = min(ny, int(np.ceil(ycenter + analysis_radius)) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        radius = np.hypot(xx - xcenter, yy - ycenter)
        source_region = radius <= max_radius
        annulus = (radius > max_radius) & (radius <= analysis_radius)
        local_median, local_std = local_stats(
            data[y0:y1, x0:x1], ~annulus
        )
        if np.isfinite(local_std) and local_std > 0:
            structure = source_region & (
                np.abs(data[y0:y1, x0:x1] - local_median) >= 2.5 * local_std
            )
            structure = binary_dilation(structure, iterations=1) & source_region
            mask[y0:y1, x0:x1] |= structure
        mask[y0:y1, x0:x1] |= radius <= core_radius
    return mask & np.isfinite(data)


def fill_source_mask(data, source_mask, fwhm, noise_seed):
    """Replace each source-mask component with local, reproducible noise."""
    cleaned = np.array(data, dtype=np.float32, copy=True)
    global_median, global_std = local_stats(data, source_mask)
    if not np.isfinite(global_median):
        global_median, global_std = 0.0, 1.0
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = 1.0

    labels, _ = label(source_mask)
    components = find_objects(labels)
    rng = np.random.default_rng(noise_seed)
    annulus_width = max(2, int(np.ceil(2.0 * fwhm)))
    ny, nx = data.shape
    for label_number, component_slice in enumerate(components, start=1):
        if component_slice is None:
            continue
        yslice, xslice = component_slice
        y0 = max(0, yslice.start - annulus_width)
        y1 = min(ny, yslice.stop + annulus_width)
        x0 = max(0, xslice.start - annulus_width)
        x1 = min(nx, xslice.stop + annulus_width)
        component = labels[y0:y1, x0:x1] == label_number
        ring = binary_dilation(component, iterations=annulus_width)
        ring &= ~binary_dilation(component, iterations=2)
        local_data = data[y0:y1, x0:x1]
        local_source_mask = source_mask[y0:y1, x0:x1]
        median, std = local_stats(local_data, local_source_mask | ~ring)
        if not np.isfinite(median):
            median, std = global_median, global_std
        if not np.isfinite(std) or std <= 0:
            std = global_std
        cleaned_region = cleaned[y0:y1, x0:x1]
        cleaned_region[component] = rng.normal(median, std, size=component.sum())
        cleaned[y0:y1, x0:x1] = cleaned_region
    return cleaned


def clean_residual_sources(residual, continuum, fwhm, threshold_sigma, noise_seed):
    """Remove continuum-matched positive and negative stellar residuals."""
    residual_positions = find_point_sources(
        residual, fwhm, threshold_sigma, signs=(1, -1)
    )
    continuum_positions = find_point_sources(
        continuum, fwhm, threshold_sigma
    )
    matched_positions = match_residual_sources(
        residual_positions, continuum_positions, max_distance=1.5 * fwhm
    )
    source_mask = make_source_mask(residual, matched_positions, fwhm)
    cleaned = fill_source_mask(residual, source_mask, fwhm, noise_seed)
    return cleaned, source_mask, len(matched_positions)


def write_output(
    filename,
    primary_header,
    image_header,
    continuum_name,
    continuum_subtracted,
    raw_continuum_subtracted,
    uncertainty,
    scaled_continuum,
    footprint,
    source_mask,
    scale,
    scale_scatter,
    scale_method,
    n_scale_pixels,
    source_cleaned,
    source_fwhm,
    source_threshold,
    n_cleaned_sources,
    noise_seed,
):
    """Write the cleaned science product and its source-removal provenance."""
    science = fits.ImageHDU(continuum_subtracted, header=image_header, name="SCI")
    science.header["CONTSUB"] = (True, "F470N continuum subtraction applied")
    science.header["CONTFIL"] = (continuum_name, "Continuum input filename")
    science.header["CONTSCL"] = (scale, "F480M continuum scale factor")
    science.header["SCLSCAT"] = (scale_scatter, "Robust clipped ratio scatter")
    science.header["SCLMETH"] = (scale_method, "Continuum scale measurement")
    science.header["NSCALE"] = (n_scale_pixels, "Sampled pixels used for scale")
    science.header["ERRMETH"] = ("QUADRATURE", "Line and scaled continuum errors")
    science.header["SRCCLEAN"] = (source_cleaned, "Continuum-matched residual stars cleaned")
    science.header["SRCMETH"] = ("DAO+F480", "Residual source detection and matching")
    science.header["SRCFWHM"] = (source_fwhm, "DAO source FWHM, pixels")
    science.header["SRCTHRS"] = (source_threshold, "DAO detection threshold, sigma")
    science.header["NSRCLEAN"] = (n_cleaned_sources, "Matched residual sources cleaned")
    science.header["NOISESED"] = (noise_seed, "Random seed for source-fill noise")

    error = fits.ImageHDU(uncertainty, header=image_header, name="ERR")
    raw = fits.ImageHDU(
        raw_continuum_subtracted, header=image_header, name="SCI_RAW"
    )
    scaled = fits.ImageHDU(
        scaled_continuum, header=image_header, name="CONT_SCALED"
    )
    coverage = fits.ImageHDU(footprint, header=image_header, name="CONT_FOOTPRINT")
    coverage.header["BUNIT"] = ("", "Dimensionless reprojected coverage")
    mask = fits.ImageHDU(source_mask.astype(np.uint8), name="SOURCE_MASK")
    mask.header["MASKED"] = (1, "Synthetic source-filled SCI pixels")

    fits.HDUList(
        [
            fits.PrimaryHDU(header=primary_header),
            science,
            error,
            raw,
            scaled,
            coverage,
            mask,
        ]
    ).writeto(filename, overwrite=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--outdir", type=Path, help="Output directory (default: DATA_DIR)"
    )
    parser.add_argument(
        "--scale", type=float, help="Use this F480M continuum scale directly"
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=10.0,
        help="Minimum S/N in each image for automatic scaling (default: 10)",
    )
    parser.add_argument(
        "--no-source-clean",
        action="store_true",
        help="Do not replace matched residual stars in the SCI extension",
    )
    parser.add_argument(
        "--source-fwhm",
        type=float,
        default=5.0,
        help="DAO source FWHM in pixels (default: 5)",
    )
    parser.add_argument(
        "--source-threshold",
        type=float,
        default=5.0,
        help="DAO detection threshold in sigma (default: 5)",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Random seed for source-fill noise (default: 0)",
    )
    args = parser.parse_args()
    if args.min_snr <= 0:
        parser.error("--min-snr must be positive")
    if args.scale is not None and args.scale <= 0:
        parser.error("--scale must be positive")
    if args.source_fwhm <= 0:
        parser.error("--source-fwhm must be positive")
    if args.source_threshold <= 0:
        parser.error("--source-threshold must be positive")

    line_file = args.data_dir / LINE_NAME
    continuum_file = args.data_dir / CONT_NAME
    outdir = args.outdir or args.data_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"line: {line_file}")
    print(f"continuum: {continuum_file}")
    line, line_error, image_header, primary_header = load_image(line_file)
    continuum, continuum_error, footprint = reproject_image_and_error(
        continuum_file, image_header
    )

    if args.scale is None:
        scale, scale_scatter, n_scale_pixels = estimate_continuum_scale(
            line, line_error, continuum, continuum_error, args.min_snr
        )
        scale_method = "PIXCLIP"
        print(
            f"continuum scale: {scale:.6g} +/- {scale_scatter:.6g} "
            f"({n_scale_pixels} sampled pixels)"
        )
    else:
        scale = args.scale
        scale_scatter = np.nan
        n_scale_pixels = 0
        scale_method = "USER"
        print(f"continuum scale: {scale:.6g} (user supplied)")

    scaled_continuum = np.asarray(scale * continuum, dtype=np.float32)
    raw_continuum_subtracted = np.asarray(line - scaled_continuum, dtype=np.float32)
    uncertainty = np.hypot(line_error, scale * continuum_error).astype(
        np.float32, copy=False
    )
    valid = np.isfinite(line) & np.isfinite(scaled_continuum) & (footprint > 0)
    raw_continuum_subtracted[~valid] = np.nan
    uncertainty[~valid] = np.nan

    if args.no_source_clean:
        continuum_subtracted = raw_continuum_subtracted.copy()
        source_mask = np.zeros(line.shape, dtype=bool)
        n_cleaned_sources = 0
    else:
        continuum_subtracted, source_mask, n_cleaned_sources = clean_residual_sources(
            raw_continuum_subtracted,
            continuum,
            args.source_fwhm,
            args.source_threshold,
            args.noise_seed,
        )
        uncertainty[source_mask] = np.nan
        print(f"source-cleaned {n_cleaned_sources} F480M-matched residual stars")
        print(f"source-filled pixels: {source_mask.sum()} ({source_mask.mean():.3%})")

    outfile = outdir / OUTPUT_NAME
    write_output(
        outfile,
        primary_header,
        image_header,
        continuum_file.name,
        continuum_subtracted,
        raw_continuum_subtracted,
        uncertainty,
        scaled_continuum,
        footprint,
        source_mask,
        scale,
        scale_scatter,
        scale_method,
        n_scale_pixels,
        not args.no_source_clean,
        args.source_fwhm,
        args.source_threshold,
        n_cleaned_sources,
        args.noise_seed,
    )
    print(f"wrote: {outfile}")


if __name__ == "__main__":
    main()
