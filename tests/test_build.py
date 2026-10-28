from datetime import datetime, timedelta, timezone
import json
import numpy as np
from windtiles.build import build, decide, frames, level_fields

RUN = datetime(2026, 10, 19, 0, tzinfo=timezone.utc)

def globe(value):
    return np.full((721, 1440), value, dtype=np.float32)

def fields(sp_pa=101_325.0):
    f = {("sp", 0, 6): globe(sp_pa)}
    for name, value in (("u", 10.0), ("v", -5.0), ("t", 220.0)):
        for level in (850, 700, 250):
            f[(name, level, 6)] = globe(value)
    for name, value in (("10u", 2.0), ("10v", 1.0), ("2t", 288.15)):
        f[(name, 0, 6)] = globe(value)
    return f

def test_frames_cover_six_to_sixty_six():
    times = frames(RUN)
    assert len(times) == 21 and times[0].hour == 6 and times[-1] == datetime(2026, 10, 21, 18, tzinfo=timezone.utc)

def test_conversion_to_kt_and_celsius():
    u, v, t = level_fields(fields(), "250", 6)
    assert abs(u[0, 0] - 19.43844) < 1e-4 and abs(v[0, 0] + 9.71922) < 1e-4 and abs(t[0, 0] + 53.15) < 1e-4
    u, _, t = level_fields(fields(), "sfc", 6)
    assert abs(u[0, 0] - 3.887688) < 1e-4 and abs(t[0, 0] - 15) < 1e-4

def test_below_ground_is_masked():
    f = fields(sp_pa=80_000.0)  # ground near 800 hPa
    assert np.isnan(level_fields(f, "850", 6)[0]).all()
    assert not np.isnan(level_fields(f, "700", 6)[0]).any()

def test_surface_level_is_never_masked():
    assert not np.isnan(level_fields(fields(sp_pa=50_000.0), "sfc", 6)[0]).any()

def test_decide():
    now = RUN + timedelta(hours=10)
    assert decide(RUN, None, now) == "build"
    assert decide(RUN, RUN - timedelta(hours=6), now) == "build"
    assert decide(RUN, RUN, now) == "up to date"
    assert decide(RUN, RUN, RUN + timedelta(hours=19)) == "stale"

def test_build_writes_every_tile_and_the_manifest(tmp_path):
    count = build(RUN, fields(), tmp_path, levels=["250"], steps=[6])
    assert count == 42
    assert (tmp_path / "wind/v1/2026101900/2026101906/250/3/3/7.png").exists()
    manifest = json.loads((tmp_path / "wind/v1/manifest.json").read_text())
    assert manifest == {"run": "2026-10-19T00:00:00Z", "frames": ["2026-10-19T06:00:00Z"]}
    assert "ECMWF" in (tmp_path / "index.html").read_text()
