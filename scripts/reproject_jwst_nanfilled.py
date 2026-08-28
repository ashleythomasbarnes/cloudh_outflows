#!/usr/bin/env python3
"""Put the nan-filled JWST images onto their three separate common grids.

MIRI, short-wave NIRCam, and long-wave NIRCam each use their own grid.  The
first alphabetically sorted image in each group is the default template.
Outputs are written below ``DATA_DIR/reprojected`` and source FITS files are
never modified.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from astropy.io import fits
from reproject import reproject_interp


DEFAULT_DATA_DIR = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Galactic/JWST/"
    "cloud_h/jwst/fits/nanfilled"
)

GROUP_PATTERNS = {
    "miri": ("_miri_",),
    "nircam-short": (
        "_nircam_clear-f115w_",
        "_nircam_clear-f182m_",
        "_nircam_clear-f200w_",
        "_nircam_f150w2-f162m_",
    ),
    "nircam-long": (
        "_nircam_clear-f356w_",
        "_nircam_clear-f480m_",
        "_nircam_f405n-f444w_",
        "_nircam_f444w-f470n_",
    ),
}


def science_hdu(hdul):
    """Return SCI when present, otherwise the first image HDU."""
    if "SCI" in hdul and hdul["SCI"].data is not None:
        return hdul["SCI"]
    for hdu in hdul:
        if hdu.data is not None and hdu.data.ndim == 2:
            return hdu
    raise ValueError("no two-dimensional science image found")


def reproject_group(instrument, files, template_file, outdir, overwrite):
    with fits.open(template_file, memmap=True) as template_hdul:
        template_hdu = science_hdu(template_hdul)
        target_header = template_hdu.header.copy()
        target_shape = template_hdu.data.shape
        template_primary = template_hdul[0].header.copy()

    print(f"\n{instrument} template: {template_file.name}")
    for filename in files:
        output = outdir / f"{filename.stem}_reprojected.fits"
        if output.exists() and not overwrite:
            print(f"Skipping existing {output.name} (use --overwrite to replace it)")
            continue

        print(f"Reprojecting {filename.name}")
        with fits.open(filename, memmap=True) as hdul:
            source_hdu = science_hdu(hdul)
            data, footprint = reproject_interp(
                (source_hdu.data, source_hdu.header),
                target_header,
                shape_out=target_shape,
            )

        data = np.asarray(data, dtype=np.float32)
        data[footprint <= 0] = np.nan
        output_header = target_header.copy()
        output_header["REPROJECT"] = (True, "Reprojected onto common instrument grid")
        output_header["REPRJTMP"] = (template_file.name, "Template image for output WCS")
        output_header["REPRJSRC"] = (filename.name, "Source image")
        fits.HDUList([
            fits.PrimaryHDU(header=template_primary),
            fits.ImageHDU(data=data, header=output_header, name="SCI"),
        ]).writeto(output, overwrite=True)
        print(f"  wrote {output}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outdir", type=Path, help="Default: DATA_DIR/reprojected")
    parser.add_argument("--miri-template", type=Path)
    parser.add_argument("--nircam-short-template", type=Path)
    parser.add_argument("--nircam-long-template", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.fits"))
    groups = {
        name: [
            file
            for file in files
            if any(pattern in file.name.lower() for pattern in patterns)
        ]
        for name, patterns in GROUP_PATTERNS.items()
    }
    missing_groups = [name for name, group in groups.items() if not group]
    if missing_groups:
        parser.error("No .fits files found for: " + ", ".join(missing_groups))

    outdir = args.outdir or args.data_dir / "reprojected"
    outdir.mkdir(parents=True, exist_ok=True)
    templates = {
        "miri": args.miri_template or groups["miri"][0],
        "nircam-short": args.nircam_short_template or groups["nircam-short"][0],
        "nircam-long": args.nircam_long_template or groups["nircam-long"][0],
    }
    for instrument in ("miri", "nircam-short", "nircam-long"):
        if not templates[instrument].is_file():
            parser.error(f"Missing {instrument} template: {templates[instrument]}")
        reproject_group(
            instrument,
            groups[instrument],
            templates[instrument],
            outdir,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
