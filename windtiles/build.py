"""Build one ECMWF run into PNG wind tiles, the manifest and index.html."""

import argparse
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from windtiles import source
from windtiles.encode import encode_tile
from windtiles.grid import LEVELS, ZOOMS, tile_grid, tile_nodes

MANIFEST_URL = "https://pilot-force-tiles.web.app/wind/v1/manifest.json"
STEPS = list(range(6, 67, 3))
MAX_AGE = timedelta(hours=18)
KT_PER_MS = 1.943844
KELVIN = 273.15

INDEX_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Pilot Force wind tiles</title>
<p>Wind tiles for Pilot Force. This service is based on data and products of the European Centre
for Medium-Range Weather Forecasts (<a href="https://www.ecmwf.int">ECMWF</a>).
Copyright © {year} ECMWF. Licence <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.
Changed: u, v in knots, T in °C, coarsened to 0.5° and 1°, 8 bits per value, masked below ground.
ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability,
or for any loss or damage arising from their use.</p>
</html>
"""


def frames(run: datetime) -> list[datetime]:
    return [run + timedelta(hours=step) for step in STEPS]


def level_fields(fields, level: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u, v (kt) and T (°C) of one level; NaN below ground on pressure levels."""
    if level == "sfc":
        u, v, t = (fields[(name, 0, step)] for name in ("10u", "10v", "2t"))
        return u * KT_PER_MS, v * KT_PER_MS, t - KELVIN
    hpa = int(level)
    u, v, t = (fields[(name, hpa, step)] for name in ("u", "v", "t"))
    ground = fields[("sp", 0, step)] < hpa * 100
    return tuple(np.where(ground, np.nan, x).astype(np.float32) for x in (u * KT_PER_MS, v * KT_PER_MS, t - KELVIN))


def build(run: datetime, fields, out: Path, levels=LEVELS, steps=None) -> int:
    """Tiles of every level and step, then the manifest and index.html; returns the tile count."""
    steps = STEPS if steps is None else steps
    root = out / "wind/v1"
    count = 0
    for index, step in enumerate(steps):
        frame = run + timedelta(hours=step)
        for level in levels:
            u, v, t = level_fields(fields, level, step)
            if index == 0:
                _summary(level, u, v, t)
            for zoom in ZOOMS:
                rows, columns = tile_grid(zoom)
                for row in range(rows):
                    for column in range(columns):
                        path = root / f"{run:%Y%m%d%H}/{frame:%Y%m%d%H}/{level}/{zoom}/{row}/{column}.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        nodes = (tile_nodes(x, zoom, row, column) for x in (u, v, t))
                        path.write_bytes(encode_tile(*nodes))
                        count += 1
    manifest = {"run": _iso(run), "frames": [_iso(run + timedelta(hours=step)) for step in steps]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (out / "index.html").write_text(INDEX_HTML.format(year=run.year))
    return count


def _summary(level: str, u, v, t) -> None:
    speed = np.hypot(u, v)
    print(
        f"{level:>4}: max {np.nanmax(speed):.0f} kt, T {np.nanmin(t):.1f} … {np.nanmax(t):.1f} °C,"
        f" no data {np.isnan(speed).mean():.1%}"
    )


def _iso(time: datetime) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")


def deployed_run(url: str = MANIFEST_URL) -> datetime | None:
    """Run of the deployed manifest; None on any network or parse error."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            run = datetime.fromisoformat(json.load(response)["run"])
        return run if run.tzinfo else run.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def decide(latest: datetime, deployed: datetime | None, now: datetime) -> str:
    if deployed is None or deployed < latest:
        return "build"
    if deployed == latest and now - deployed > MAX_AGE:
        return "stale"
    return "up to date"


def main(argv=None) -> int:
    from ecmwf.opendata import Client

    parser = argparse.ArgumentParser(prog="windtiles.build")
    parser.add_argument("--out", type=Path, required=True, help="site directory")
    parser.add_argument("--work", type=Path, required=True, help="GRIB download directory, outside the repository")
    parser.add_argument("--force", action="store_true", help="build without comparing with the deployed run")
    args = parser.parse_args(argv)

    client = Client(source=source.SOURCE)
    run = source.latest_run(client)
    if not args.force:
        deployed = deployed_run()
        decision = decide(run, deployed, datetime.now(timezone.utc))
        if decision != "build":
            if decision == "stale":
                print(f"deployed run {_iso(deployed)} older than 18 h, no newer run with step 66")
            else:
                print(f"run {_iso(run)} up to date")
            _output(changed=False)
            return 1 if decision == "stale" else 0

    print(f"building run {_iso(run)}")
    fields = source.read_fields(source.download(client, run, STEPS, args.work))
    print(f"{build(run, fields, args.out)} tiles in {args.out}")
    _output(changed=True)
    return 0


def _output(changed: bool) -> None:
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as file:
            file.write(f"changed={'true' if changed else 'false'}\n")


if __name__ == "__main__":
    raise SystemExit(main())
