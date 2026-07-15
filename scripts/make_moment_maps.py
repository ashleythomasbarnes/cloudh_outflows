from pathlib import Path

from astropy import units as u
from astropy.io import fits
from spectral_cube_plus import SpectralCubePlus


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CUBE_PATTERN = "*.cube.I.*pbcor.fits"


def make_moment_maps(cube_path):
    """Create RMS, moment-0, and maximum maps for one spectral cube."""
    print(f"Processing {cube_path.name}")

    cube = SpectralCubePlus.read(cube_path)
    cube.allow_huge_operations = True
    cube = cube.with_spectral_unit(u.km / u.s, velocity_convention="radio")
    cube = cube.to(u.K)

    rms = cube.get_rms_auto()
    rms_map = fits.PrimaryHDU(rms.value, header=cube.header)
    rms_map.writeto(cube_path.with_name(f"{cube_path.stem}_rms.fits"), overwrite=True)

    cube.get_expmask(hthresh=3, lthresh=2)
    moment0 = cube.masked.moment0()
    maximum = cube.masked.max(axis=0)

    moment0.write(
        cube_path.with_name(f"{cube_path.stem}_moment0.fits"), overwrite=True
    )
    maximum.write(
        cube_path.with_name(f"{cube_path.stem}_maximum.fits"), overwrite=True
    )


def main():
    cube_paths = sorted(DATA_DIR.glob(CUBE_PATTERN))
    if not cube_paths:
        raise FileNotFoundError(
            f"No input cubes matching {CUBE_PATTERN!r} found in {DATA_DIR}"
        )

    for cube_path in cube_paths:
        make_moment_maps(cube_path)


if __name__ == "__main__":
    main()
