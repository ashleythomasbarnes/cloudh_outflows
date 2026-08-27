#!/usr/bin/env python3
"""Make the Pa-alpha and Br-alpha JWST continuum-subtracted maps."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from reproject import reproject_interp


DEFAULT_DATA_DIR = Path(
    "/Users/abarnes/Library/CloudStorage/Dropbox/Data/Galactic/JWST/"
    "cloud_h/cloud_H_JWST/NIRCam/12-05-2026"
)

PAIRS = {
    "PaA_F182M_minus_F200W": {
        "line": "jw07230-o004_t003_nircam_clear-f182m_i2d.fits",
        "cont": "jw07230-o004_t003_nircam_clear-f200w_i2d.fits",
    },
    # The F405N observation uses F444W as the paired NIRCam filter element;
    # the separate continuum image available here is F356W.
    "BrA_F405N_minus_F356W": {
        "line": "jw07230-o004_t003_nircam_f405n-f444w_i2d.fits",
        "cont": "jw07230-o004_t003_nircam_clear-f356w_i2d.fits",
    },
}


def get_data_header(filename, ext="SCI"):
    with fits.open(filename, memmap=True) as hdul:
        data = np.asarray(hdul[ext].data, dtype=np.float32)
        header = hdul[ext].header.copy()
        primary = hdul[0].header.copy()
    return data, header, primary


def reproject_to_match(input_file, target_header, ext="SCI"):
    with fits.open(input_file, memmap=True) as hdul:
        reproj, footprint = reproject_interp(
            (hdul[ext].data, hdul[ext].header), target_header
        )
    reproj = np.asarray(reproj, dtype=np.float32)
    reproj[footprint <= 0] = np.nan
    return reproj


def robust_continuum_scale(line, cont, nsigma=3.0):
    """Estimate the multiplicative continuum scale from valid positive pixels."""
    mask = np.isfinite(line) & np.isfinite(cont) & (cont > 0) & (line > 0)
    ratio = sigma_clip(
        line[mask] / cont[mask], sigma=nsigma, maxiters=5, masked=False
    )
    scale = float(np.nanmedian(ratio))
    scatter = float(1.4826 * np.nanmedian(np.abs(ratio - scale)))
    return scale, scatter


def process_pair(name, files, data_dir, outdir):
    line_file = data_dir / files["line"]
    cont_file = data_dir / files["cont"]
    print(f"\nProcessing {name}")
    print(f"  line: {line_file}")
    print(f"  cont: {cont_file}")

    line, line_header, primary_header = get_data_header(line_file)
    cont = reproject_to_match(cont_file, line_header)
    scale, scale_scatter = robust_continuum_scale(line, cont)
    cont_scaled = scale * cont
    line_contsub = line - cont_scaled

    print(f"  continuum scale = {scale:.6g}")
    print(f"  robust scatter  = {scale_scatter:.6g}")

    hdu_sci = fits.ImageHDU(line_contsub, header=line_header, name="SCI")
    hdu_sci.header["CONTSUB"] = True
    hdu_sci.header["CONTFIL"] = cont_file.name
    hdu_sci.header["CONTSCL"] = (scale, "Continuum scale factor")
    hdu_sci.header["SCLSCAT"] = (scale_scatter, "Robust scale scatter")

    outfile = outdir / f"{name}_contsub.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(header=primary_header),
            hdu_sci,
            fits.ImageHDU(cont_scaled, header=line_header, name="CONT_SCALED"),
            fits.ImageHDU(line_contsub, header=line_header, name="LINE_CONT_SUB"),
        ]
    ).writeto(outfile, overwrite=True)
    print(f"  wrote: {outfile}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Output directory (default: DATA_DIR/contsub)",
    )
    args = parser.parse_args()
    outdir = args.outdir or args.data_dir / "contsub"
    outdir.mkdir(parents=True, exist_ok=True)

    for name, files in PAIRS.items():
        process_pair(name, files, args.data_dir, outdir)


if __name__ == "__main__":
    main()
