"""pytest 공용 픽스처.

- 기본 실행(-m "not api")은 재현 모드(REPLAY_OFFLINE=1)로 고정한다. 캐시에 없는 LLM·검색 호출은 과금 없이 바로 실패한다.
- 테스트마다 설정 캐시(get_config)를 비우고 시작한다. 설정을 바꿔야 하면 set_cfg 픽스처를 쓴다(끝나면 되돌림).
"""
from __future__ import annotations

import os

import pytest

from core.config import get_config

_MISSING = object()


def pytest_configure(config):
    markexpr = (config.getoption("markexpr", default="") or "").replace(" ", "")
    if markexpr != "api":  # `-m api` 로 명시했을 때만 실제 API 호출을 허용
        os.environ["REPLAY_OFFLINE"] = "1"


@pytest.fixture(autouse=True)
def _fresh_config():
    """테스트 사이에 설정이 새지 않게 get_config 캐시를 앞뒤로 비운다."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture
def set_cfg():
    """set_cfg("workflow.calibrate", True) 처럼 점 표기 경로의 설정값을 이 테스트 동안만 바꾼다.
    키에 점이 있으면(모델명 등) 경로를 튜플로 준다: set_cfg(("models", "price_per_mtok", 모델명), [1, 2]).

    get_config() 가 돌려주는 캐시 객체의 원본 dict 를 직접 고치므로, 코드가 get_config() 를 다시 불러도 바뀐 값이 보인다.
    (cfg.workflow 처럼 점 표기로 꺼낸 하위 Config 는 사본이라 거기에 쓰면 반영되지 않는다.)"""
    saved: list[tuple[dict, str, object]] = []

    def _set(key: str | tuple[str, ...], value) -> None:
        *parents, leaf = key.split(".") if isinstance(key, str) else key
        d = get_config()
        for k in parents:
            d = dict.__getitem__(d, k)
        saved.append((d, leaf, d.get(leaf, _MISSING)))
        d[leaf] = value

    yield _set
    for d, leaf, old in reversed(saved):
        if old is _MISSING:
            d.pop(leaf, None)
        else:
            d[leaf] = old
