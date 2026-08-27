#!/usr/bin/env python3
"""Put the nan-filled JWST NIRCam and MIRI images onto separate common grids.

The first alphabetically sorted image from each instrument is the default
template.  Outputs are written below ``DATA_DIR/reprojected`` and the source
FITS files are never modified.
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

    print(f"\n{instrument.upper()} template: {template_file.name}")
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
    parser.add_argument("--nircam-template", type=Path)
    parser.add_argument("--miri-template", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.data_dir.glob("*.fits"))
    groups = {
        "nircam": [file for file in files if "_nircam_" in file.name.lower()],
        "miri": [file for file in files if "_miri_" in file.name.lower()],
    }
    if not groups["nircam"] or not groups["miri"]:
        parser.error("Need at least one NIRCam and one MIRI .fits file")

    outdir = args.outdir or args.data_dir / "reprojected"
    outdir.mkdir(parents=True, exist_ok=True)
    templates = {
        "nircam": args.nircam_template or groups["nircam"][0],
        "miri": args.miri_template or groups["miri"][0],
    }
    for instrument in ("nircam", "miri"):
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
