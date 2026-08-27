from pathlib import Path

from astropy import units as u
from astropy.io import fits
from spectral_cube_plus import SpectralCubePlus


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VELOCITY_MIN = -50 * u.km / u.s
VELOCITY_MAX = 150 * u.km / u.s

LINE_CUBES = {
    "member.uid___A001_X3819_X177._035.522-00.274__sci.spw25.cube.I.selfcal.pbcor.fits": (
        "CO_3-2",
        345.79599 * u.GHz,
    ),
    "member.uid___A001_X3819_X177._035.522-00.274__sci.spw27.cube.I.selfcal.pbcor.fits": (
        "SiO_8-7",
        347.3306 * u.GHz,
    ),
    "member.uid___A001_X3621_X2808.Clump_H6_sci.spw31.cube.I.pbcor.fits": (
        "SO_6(5)-5(4)",
        219.9494 * u.GHz,
    ),
    "member.uid___A001_X3621_X2808.Clump_H6_sci.spw33.cube.I.pbcor.fits": (
        "13CO_2-1",
        220.3987 * u.GHz,
    ),
    "member.uid___A001_X3621_X2808.Clump_H6_sci.spw43.cube.I.pbcor.fits": (
        "12CO_2-1",
        230.5380 * u.GHz,
    ),
    "member.uid___A001_X879_X395.CloudH_sci.spw27.cube.I.pbcor.fits": (
        "SiO_2-1",
        86.8470 * u.GHz,
    ),
    "member.uid___A001_X879_X395.CloudH_sci.spw33.cube.I.pbcor.fits": (
        "CH3OH_2(0,2)-1(0,1)++",
        96.74138 * u.GHz,
    ),
    "member.uid___A001_X879_X395.CloudH_sci.spw37.cube.I.pbcor.fits": (
        "CS_2-1",
        97.9810 * u.GHz,
    ),
}


def output_path(cube_path, line_name, product):
    return cube_path.with_name(f"{cube_path.stem}_{line_name}_{product}.fits")


def make_moment_maps(cube_path, line_name, rest_frequency):
    """Cut out one spectral line and create its masked moment maps."""
    print(
        f"Processing {line_name} at {rest_frequency.to_value(u.GHz):.5f} GHz "
        f"from {cube_path.name}"
    )

    cube_plus = SpectralCubePlus.read(cube_path)
    cube_plus.allow_huge_operations = True
    cube_plus = cube_plus.with_spectral_unit(
        u.km / u.s,
        velocity_convention="radio",
        rest_value=rest_frequency,
    )
    cube_plus = cube_plus.to(u.K)
    cube_plus = cube_plus.spectral_slab(VELOCITY_MIN, VELOCITY_MAX)
    cube_plus.allow_huge_operations = True

    cube_plus.write(
        output_path(cube_path, line_name, "cube_-50_to_150_kms"),
        overwrite=True,
    )

    rms = cube_plus.get_rms_auto()
    rms_hdu = fits.PrimaryHDU(rms.value, header=cube_plus.rms_hdu.header)
    rms_hdu.writeto(
        output_path(cube_path, line_name, "rms"),
        overwrite=True,
    )

    cube_plus.get_expmask(hthresh=3, lthresh=2)
    moment0 = cube_plus.masked.moment0()
    cube_plus.masked.mom0 = moment0
    moment0err = cube_plus.masked.moment0err()
    moment1 = cube_plus.masked.moment1()
    moment2 = cube_plus.masked.linewidth_sigma()
    maximum = cube_plus.masked.max(axis=0)

    products = {
        "moment0": moment0,
        "moment0err": moment0err,
        "moment1": moment1,
        "moment2": moment2,
        "maximum": maximum,
        "moment0_over_moment0err": moment0 / moment0err,
        "maximum_over_rms": maximum / rms,
    }
    for product_name, product in products.items():
        product.write(
            output_path(cube_path, line_name, product_name), overwrite=True
        )


def main():
    missing_paths = [
        DATA_DIR / name for name in LINE_CUBES if not (DATA_DIR / name).is_file()
    ]
    if missing_paths:
        missing_names = "\n".join(f"  {path.name}" for path in missing_paths)
        raise FileNotFoundError(f"Missing input cubes in {DATA_DIR}:\n{missing_names}")

    for filename, (line_name, rest_frequency) in LINE_CUBES.items():
        make_moment_maps(DATA_DIR / filename, line_name, rest_frequency)


if __name__ == "__main__":
    main()
