"""config.yaml 과 .env 를 한곳에서 읽는다."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """점 표기(cfg.models.generator)로도 읽을 수 있는 dict."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> Config:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(path) if path else ROOT / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def require_keys() -> None:
    """API 키는 실제로 API 를 부를 때만 필요하다 (재현용 캐시만으로 실행할 때는 없어도 된다).
    검색 키는 Serper·Tavily 중 하나만 있어도 된다."""
    missing = [] if os.getenv("OPENAI_API_KEY") else ["OPENAI_API_KEY"]
    if not any(os.getenv(k) for k in ("SERPER_API_KEY", "TAVILY_API_KEY")):
        missing.append("검색 키(SERPER_API_KEY 또는 TAVILY_API_KEY)")
    if missing:
        raise RuntimeError(f"캐시에 없는 호출이라 {', '.join(missing)} 가 필요합니다. 저장소 루트에 .env 를 만들어 OPENAI_API_KEY 와 SERPER_API_KEY(또는 TAVILY_API_KEY)를 넣으세요.")


def run_date() -> str:
    """평가 기준일. 재현용 캐시(replay/run_meta.json)가 있으면 그 날짜로 고정해 24개월 판정·조회일이 달라지지 않게 한다."""
    import json
    from datetime import datetime

    meta = ROOT / get_config().cache.dir / "run_meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))["run_date"]
    return datetime.now().strftime("%Y-%m-%d")


def path(rel: str) -> Path:
    """프로젝트 루트 기준 상대 경로를 절대 경로로."""
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_segment(segment_id: str) -> dict:
    """세부 분야 id → config 의 분야 정보. LLM 이 여러 id 를 붙여 쓰면 첫 번째 유효한 id 를 쓴다."""
    segs = get_config().domain.segments
    for part in str(segment_id).replace("/", ",").split(","):
        for s in segs:
            if s["id"] == part.strip():
                return s
    return segs[0]
