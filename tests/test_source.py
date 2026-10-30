from datetime import datetime, timezone
from types import SimpleNamespace
import eccodes
import numpy as np
import pytest
from windtiles import source
from windtiles.source import read_fields

RUN = datetime(2026, 10, 19, 0, tzinfo=timezone.utc)

def grib(path, ni=1440, nj=721, step_deg=0.25, first_lat=90.0, keys=None, values=None):
    handle = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
    eccodes.codes_set_key_vals(handle, {
        "shortName": "u", "level": 250, "step": 6, "Ni": ni, "Nj": nj,
        "iDirectionIncrementInDegrees": step_deg, "jDirectionIncrementInDegrees": step_deg,
        "latitudeOfFirstGridPointInDegrees": first_lat,
        "latitudeOfLastGridPointInDegrees": first_lat - (nj - 1) * step_deg,
        "longitudeOfFirstGridPointInDegrees": 180.0,
        "longitudeOfLastGridPointInDegrees": 180.0 + (ni - 1) * step_deg - 360,
        **(keys or {}),
    })
    rows = np.repeat(np.arange(nj, dtype=np.float64)[:, None] / 10, ni, axis=1)  # value = GRIB row / 10
    eccodes.codes_set_values(handle, rows.ravel() if values is None else values)
    with open(path, "wb") as file:
        eccodes.codes_write(handle, file)
    eccodes.codes_release(handle)
    return path

def test_reads_and_normalizes_a_global_field(tmp_path):
    fields = read_fields([grib(tmp_path / "pl.grib2")])
    globe = fields[("u", 250, 6)]
    assert globe.shape == (721, 1440)
    assert globe[0, 0] == pytest.approx(72, abs=0.1)   # south pole is the last GRIB row
    assert globe[720, 0] == pytest.approx(0, abs=0.1)

@pytest.mark.parametrize("grid", [{"ni": 720, "step_deg": 0.5}, {"nj": 361, "step_deg": 0.5}, {"first_lat": 89.75}])
def test_rejects_a_foreign_grid(tmp_path, grid):
    with pytest.raises(ValueError, match="u 250 6"):
        read_fields([grib(tmp_path / "bad.grib2", **grid)])

@pytest.mark.parametrize("keys", [{"iScansNegatively": 1}, {"jScansPositively": 1}])
def test_rejects_a_flipped_scan(tmp_path, keys):
    with pytest.raises(ValueError, match="u 250 6: scanned"):
        read_fields([grib(tmp_path / "flip.grib2", keys=keys)])

def test_rejects_missing_values(tmp_path):
    values = np.zeros(1440 * 721)
    values[100] = 9999
    with pytest.raises(ValueError, match="u 250 6: has missing"):
        read_fields([grib(tmp_path / "gap.grib2", keys={"bitmapPresent": 1, "missingValue": 9999}, values=values)])

MUCAPE = {"shortName": "mucape", "typeOfLevel": "mostUnstableParcel"}  # level 250 of the sample hides it otherwise

@pytest.mark.parametrize("keys, valid, value", [
    ({}, 10.0, 150.0), ({}, 10.0, -151.0), ({"shortName": "t"}, 250.0, 140.0),
    ({"shortName": "gh"}, 10_000.0, 30_000), ({"shortName": "r"}, 50.0, 200),
    ({"shortName": "r"}, 50.0, -11), (MUCAPE, 500.0, 25_000),
])
def test_rejects_a_value_out_of_range(tmp_path, keys, valid, value):
    values = np.full(1440 * 721, valid)
    values[100] = value
    with pytest.raises(ValueError, match="out of range"):
        read_fields([grib(tmp_path / "range.grib2", keys=keys, values=values)])

def test_accepts_slightly_negative_humidity(tmp_path):  # IFS sends r down to −5.6 %
    values = np.full(1440 * 721, 50.0)
    values[100] = -5.6
    fields = read_fields([grib(tmp_path / "r.grib2", keys={"shortName": "r"}, values=values)])
    assert fields[("r", 250, 6)].min() == pytest.approx(-5.6, abs=0.01)

def test_download_asks_levels_and_surface_with_step_3(tmp_path):
    calls = []
    client = SimpleNamespace(retrieve=lambda **kw: calls.append(kw))
    source.download(client, RUN, [6, 9], tmp_path)
    pl, sfc = calls
    assert pl["levelist"] == source.PL_LEVELS and pl["param"] == source.PL_PARAMS and pl["step"] == [6, 9]
    assert sfc["param"] == source.SFC_PARAMS and sfc["step"] == [3, 6, 9]
