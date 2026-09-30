"""국민연금 조회 스냅샷의 동시 쓰기 (적격성 관문이 후보 4곳을 스레드로 동시에 검사한다).

잠금 없이 '읽기 → 조회 → 통째로 쓰기'를 하면 쓰기가 겹쳐 JSON 뒤에 꼬리가 붙는다(JSONDecodeError: Extra data).
- 여러 스레드가 동시에 조회해도 스냅샷은 온전한 JSON 이고 모든 조회 결과가 남는다
- 예전 버전이 깨뜨린 스냅샷은 앞의 온전한 JSON 을 살리고 원본은 백업한다
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

import tools.nps as nps


@pytest.fixture
def snap_file(tmp_path, monkeypatch):
    f = tmp_path / "snapshots" / "nps_lookup.json"
    monkeypatch.setattr(nps, "_snapshot_file", lambda: f)
    names = [f"테스트농업{i:02d}" for i in range(32)]
    df = pd.DataFrame({"ym": "2026-08", "name": names, "brn6": [f"{i:06d}" for i in range(32)], "status": "1",
                       "corp": "1", "industry": "농업", "first_date": "2022-01-01", "withdraw_date": "",
                       "members": "5", "new": "0", "lost": "0"})
    df["key"] = df["name"].map(nps._norm)
    barrier = threading.Barrier(8, timeout=1)

    def slow_table():  # 조회 사이에 다른 스레드가 끼어들 틈을 크게 만든다
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return df, "2026-08"

    monkeypatch.setattr(nps, "_table", slow_table)
    return f, names


def test_concurrent_lookups_keep_snapshot_valid(snap_file):
    f, names = snap_file
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda n: nps.lookup([n]), names))
    assert all(r["status"] == "matched" for r in results)
    snap = json.loads(f.read_text(encoding="utf-8"))          # Extra data 없이 읽힌다
    assert len(snap) == len(names)                            # 어느 스레드의 결과도 덮어써지지 않았다
    assert not list(f.parent.glob("*.tmp"))                   # 임시 파일이 남지 않는다


def test_corrupt_snapshot_is_salvaged(snap_file):
    f, names = snap_file
    f.parent.mkdir(parents=True, exist_ok=True)
    good = {nps._norm(names[0]): {"status": "matched", "matches": [], "ym": "2026-08"}}
    f.write_text(json.dumps(good, ensure_ascii=False, indent=1) + '\n "꼬리": {"status": "not_found"}\n}', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        json.loads(f.read_text(encoding="utf-8"))            # 팀원 실행에서 난 것과 같은 상태
    assert nps.lookup([names[0]]) == good[nps._norm(names[0])]  # 살린 조회 결과를 그대로 쓴다
    assert json.loads(f.read_text(encoding="utf-8")) == good  # 스냅샷은 다시 온전하다
    assert len(list(f.parent.glob("nps_lookup.corrupt-*.json"))) == 1  # 원본은 백업
