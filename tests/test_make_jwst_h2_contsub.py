import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


pytest.importorskip("photutils")
pytest.importorskip("reproject")

SCRIPT = Path(__file__).parents[1] / "scripts" / "make_jwst_h2_contsub.py"
SPEC = importlib.util.spec_from_file_location("make_jwst_h2_contsub", SCRIPT)
CONT_SUB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONT_SUB)


def add_gaussian(data, xcenter, ycenter, amplitude, sigma):
    y, x = np.indices(data.shape)
    data += amplitude * np.exp(
        -((x - xcenter) ** 2 + (y - ycenter) ** 2) / (2.0 * sigma**2)
    )


def test_clean_residual_sources_preserves_line_only_knot(tmp_path):
    rng = np.random.default_rng(42)
    residual = rng.normal(0.0, 0.1, size=(128, 128)).astype(np.float32)
    continuum = rng.normal(0.0, 0.1, size=(128, 128)).astype(np.float32)

    add_gaussian(residual, 30, 30, 10.0, 1.2)
    add_gaussian(residual, 70, 70, -10.0, 1.2)
    add_gaussian(residual, 100, 100, 10.0, 1.2)
    add_gaussian(continuum, 30, 30, 20.0, 1.2)
    add_gaussian(continuum, 70, 70, 20.0, 1.2)

    cleaned, source_mask, n_sources = CONT_SUB.clean_residual_sources(
        residual, continuum, fwhm=3.0, threshold_sigma=5.0, noise_seed=0
    )

    assert n_sources == 2
    assert source_mask[30, 30]
    assert source_mask[70, 70]
    assert not source_mask[100, 100]
    assert cleaned[30, 30] != residual[30, 30]
    assert cleaned[70, 70] != residual[70, 70]
    assert cleaned[100, 100] == residual[100, 100]

    uncertainty = np.full(residual.shape, 0.2, dtype=np.float32)
    uncertainty[source_mask] = np.nan
    header = fits.Header()
    header["BUNIT"] = "MJy/sr"
    outfile = tmp_path / "cleaned.fits"
    CONT_SUB.write_output(
        outfile,
        fits.Header(),
        header,
        "continuum.fits",
        cleaned,
        residual,
        uncertainty,
        continuum,
        np.ones(residual.shape, dtype=np.float32),
        source_mask,
        1.0,
        0.1,
        "PIXCLIP",
        100,
        True,
        3.0,
        5.0,
        n_sources,
        0,
    )

    with fits.open(outfile) as hdul:
        assert [hdu.name for hdu in hdul] == [
            "PRIMARY",
            "SCI",
            "ERR",
            "SCI_RAW",
            "CONT_SCALED",
            "CONT_FOOTPRINT",
            "SOURCE_MASK",
        ]
        assert hdul["SCI"].header["SRCCLEAN"]
        assert hdul["SCI"].header["NSRCLEAN"] == 2
        assert np.isnan(hdul["ERR"].data[source_mask]).all()
        np.testing.assert_allclose(hdul["SCI_RAW"].data, residual)


def test_tiled_detection_does_not_duplicate_boundary_source():
    old_tile_size = CONT_SUB.TILE_SIZE
    CONT_SUB.TILE_SIZE = 32
    try:
        rng = np.random.default_rng(1)
        residual = rng.normal(0.0, 0.1, size=(80, 80)).astype(np.float32)
        continuum = rng.normal(0.0, 0.1, size=(80, 80)).astype(np.float32)
        add_gaussian(residual, 32, 40, 10.0, 1.2)
        add_gaussian(continuum, 32, 40, 20.0, 1.2)
        _, source_mask, n_sources = CONT_SUB.clean_residual_sources(
            residual, continuum, fwhm=3.0, threshold_sigma=5.0, noise_seed=0
        )
    finally:
        CONT_SUB.TILE_SIZE = old_tile_size

    assert n_sources == 1
    assert source_mask[40, 32]
