"""ECMWF open data IFS 0.25°: latest run, download, GRIB2 → normalized globes."""

from datetime import datetime, timezone
from pathlib import Path

import eccodes
import numpy as np

from windtiles.grid import NI, NJ, STEP_DEG, normalize

SOURCE = "google"  # ECMWF open data mirror; AWS answered 503 Slow Down on 2026-10-28
PL_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100]
PL_PARAMS = ["u", "v", "t", "gh", "r"]
SFC_PARAMS = ["10u", "10v", "2t", "sp", "mucape", "cp"]
WINDS = {"u", "v", "10u", "10v"}
WIND_LIMIT = 150  # m/s; |wind| at or past it is a broken field
RANGES = {  # K, K, Pa, m, %, J/kg, m
    "t": (150, 350),
    "2t": (150, 350),
    "sp": (30_000, 115_000),
    "gh": (-1_000, 25_000),
    "r": (-1, 160),
    "mucape": (-1, 20_000),
    "cp": (-1e-6, 2),
}


def latest_run(client) -> datetime:
    """Latest oper run that already has step 66, UTC."""
    run = client.latest(stream="oper", type="fc", step=66, param="u", levelist=150)
    return run.replace(tzinfo=timezone.utc)


def download(client, run: datetime, steps: list[int], directory: Path) -> list[Path]:
    """Pressure levels and surface fields of one run into two GRIB2 files; surface also at step 3 for cp."""
    directory.mkdir(parents=True, exist_ok=True)
    common = dict(date=run.strftime("%Y%m%d"), time=run.hour, stream="oper", type="fc")
    pl, sfc = directory / "pl.grib2", directory / "sfc.grib2"
    client.retrieve(**common, step=steps, param=PL_PARAMS, levelist=PL_LEVELS, target=str(pl))
    client.retrieve(**common, step=[3] + steps, param=SFC_PARAMS, target=str(sfc))
    return [pl, sfc]


def read_fields(paths: list[Path]) -> dict[tuple[str, int, int], np.ndarray]:
    """(shortName, level, step) → normalized (721, 1440) globe; level 0 for surface fields."""
    fields = {}
    for path in paths:
        with open(path, "rb") as file:
            while (handle := eccodes.codes_grib_new_from_file(file)) is not None:
                try:
                    key, globe = _read_message(handle)
                finally:
                    eccodes.codes_release(handle)
                fields[key] = globe
    return fields


def _read_message(handle) -> tuple[tuple[str, int, int], np.ndarray]:
    get = lambda name: eccodes.codes_get(handle, name)
    pressure = get("typeOfLevel") == "isobaricInhPa"
    key = (get("shortName"), get("level") if pressure else 0, eccodes.codes_get_long(handle, "step"))
    first_lat = get("latitudeOfFirstGridPointInDegrees")
    if (
        get("Ni") != NI
        or get("Nj") != NJ
        or get("iDirectionIncrementInDegrees") != STEP_DEG
        or get("jDirectionIncrementInDegrees") != STEP_DEG
        or abs(first_lat) != 90
    ):
        raise ValueError(f"{' '.join(map(str, key))}: not a global {STEP_DEG}° {NI} × {NJ} grid from a pole")
    if get("iScansNegatively") != 0 or get("jScansPositively") != (1 if first_lat < 0 else 0):
        raise ValueError(f"{' '.join(map(str, key))}: scanned against the first point")
    if get("bitmapPresent"):
        raise ValueError(f"{' '.join(map(str, key))}: has missing values")
    values = eccodes.codes_get_values(handle).reshape(NJ, NI)
    if not np.isfinite(values).all():
        raise ValueError(f"{' '.join(map(str, key))}: non-finite values")
    low, high = RANGES.get(key[0], (-np.inf, np.inf))
    if (key[0] in WINDS and np.abs(values).max() >= WIND_LIMIT) or values.min() < low or values.max() > high:
        raise ValueError(f"{' '.join(map(str, key))}: values out of range")
    return key, normalize(values, first_lat, get("longitudeOfFirstGridPointInDegrees"))
