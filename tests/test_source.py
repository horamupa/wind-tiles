import eccodes
import numpy as np
import pytest
from windtiles.source import read_fields

def grib(path, ni=1440, nj=721, step_deg=0.25, first_lat=90.0):
    handle = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
    eccodes.codes_set_key_vals(handle, {
        "shortName": "u", "level": 250, "step": 6, "Ni": ni, "Nj": nj,
        "iDirectionIncrementInDegrees": step_deg, "jDirectionIncrementInDegrees": step_deg,
        "latitudeOfFirstGridPointInDegrees": first_lat,
        "latitudeOfLastGridPointInDegrees": first_lat - (nj - 1) * step_deg,
        "longitudeOfFirstGridPointInDegrees": 180.0,
        "longitudeOfLastGridPointInDegrees": 180.0 + (ni - 1) * step_deg - 360,
    })
    rows = np.repeat(np.arange(nj, dtype=np.float64)[:, None], ni, axis=1)  # value = GRIB row
    eccodes.codes_set_values(handle, rows.ravel())
    with open(path, "wb") as file:
        eccodes.codes_write(handle, file)
    eccodes.codes_release(handle)
    return path

def test_reads_and_normalizes_a_global_field(tmp_path):
    fields = read_fields([grib(tmp_path / "pl.grib2")])
    globe = fields[("u", 250, 6)]
    assert globe.shape == (721, 1440)
    assert globe[0, 0] == pytest.approx(720, abs=0.1)   # south pole is the last GRIB row
    assert globe[720, 0] == pytest.approx(0, abs=0.1)

@pytest.mark.parametrize("grid", [{"ni": 720, "step_deg": 0.5}, {"nj": 361, "step_deg": 0.5}, {"first_lat": 89.75}])
def test_rejects_a_foreign_grid(tmp_path, grid):
    with pytest.raises(ValueError, match="u 250 6"):
        read_fields([grib(tmp_path / "bad.grib2", **grid)])
