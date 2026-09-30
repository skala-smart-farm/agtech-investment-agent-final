"""캐시 보존 동결 검사 (v2 계약 C0).

발굴·적격성 관문의 LLM·검색 캐시 키는 아래 파일과 설정값에서 만들어진다. 하나라도 바뀌면 재현용 캐시가 맞지 않아
관문 판정을 다시 부르고(유료) 후보가 달라질 수 있다. 그래서 v1-safe 태그와 비교한다.
- FROZEN_FILES: 한 글자도 바꾸지 않는다 (git diff v1-safe 가 비어 있어야 함)
- tools/fetch.py: 기존 정의 전부, tools/sources.py: 근거 등록에 쓰는 기존 정의만 동결 (새 함수 추가는 허용)
- config.yaml: 기존 키 값은 그대로 (새 키 추가만 허용). v2 병합 때 지울 v1 판단 규칙 3키만 예외
"""
from __future__ import annotations

import ast
import subprocess

import pytest
import yaml

from core.config import ROOT

TAG = "v1-safe"
FROZEN_FILES = [
    "agents/discovery.py", "agents/eligibility.py",
    "prompts/rag_grade.md", "prompts/rag_rewrite.md", "prompts/discovery_extract.md", "prompts/eligibility.md",
    "prompts/market_decompose.md", "prompts/competition_plan.md",
    "core/llm.py", "core/prompts.py", "tools/grounding.py", "rag/index.py",
]
# 근거 등록(add_web·add_doc)과 그 결과(사이트명·제목·날짜·기자)는 관문 프롬프트에 들어가므로 동결.
# REFERENCE 표기(format_reference 등)는 캐시 키가 아니라서 수정 가능하다.
SOURCES_FROZEN = [
    "SITE_NAMES", "_site", "SITE_TAIL", "_clean_title", "_date", "_date_from", "NOT_REPORTER", "_author", "today",
    "SourceRegistry.__init__", "SourceRegistry._id", "SourceRegistry._find", "SourceRegistry.add_web",
    "SourceRegistry.add_doc", "SourceRegistry.get", "SourceRegistry.text", "SourceRegistry.brief",
]
# v2 병합 때 지울 v1 판단 규칙 키 (지워도 되고, 남아 있으면 값이 같아야 함)
CONFIG_REMOVABLE = ("decision.invest_threshold", "decision.min_dimension_ratio", "decision.max_unknown_ratio")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)


@pytest.fixture(scope="module", autouse=True)
def _need_tag():
    if _git("rev-parse", "--verify", "-q", f"{TAG}^{{commit}}").returncode != 0:
        pytest.skip(f"git 태그 {TAG} 가 없는 clone (동결 검사는 개발 저장소에서만)")


def _at_tag(rel: str) -> str:
    out = _git("show", f"{TAG}:{rel}")
    assert out.returncode == 0, out.stderr.decode()
    return out.stdout.decode("utf-8")


def _defs(src: str) -> dict[str, str]:
    """모듈 최상위 함수·상수와 클래스 메서드 → AST 덤프 (주석·공백 변경은 무시하고 코드·docstring 변경은 잡는다)."""
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[f"{node.name}.{sub.name}"] = ast.dump(sub)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                if isinstance(t, ast.Name):
                    out[t.id] = ast.dump(node)
    return out


def _leaves(d: dict, prefix: str = "") -> dict[str, object]:
    """중첩 dict → {점 표기 경로: 값}. 목록은 값 전체로 비교한다."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_leaves(v, key + "."))
        else:
            out[key] = v
    return out


def _removable(key: str) -> bool:
    return any(key == r or key.startswith(r + ".") for r in CONFIG_REMOVABLE)


def test_frozen_files_unchanged():
    changed = _git("diff", "--name-only", TAG, "--", *FROZEN_FILES).stdout.decode().split()
    assert not changed, f"캐시 키 파일이 {TAG} 와 다름: {changed}"


def test_fetch_existing_definitions_unchanged():
    old, new = _defs(_at_tag("tools/fetch.py")), _defs((ROOT / "tools/fetch.py").read_text(encoding="utf-8"))
    changed = [k for k, v in old.items() if new.get(k) != v]
    assert not changed, f"tools/fetch.py 기존 정의가 바뀜(추가만 허용): {changed}"


# 허용한 유일한 변경: 새 근거 id 를 _id 대신 _new_id 로 만든다(해시 충돌 때만 id 가 길어짐, 충돌이 없으면 _id 와 같다).
# 재현용 캐시 실행(본 실행 586개·보정 898개 id)에는 충돌이 0건이라 프롬프트 속 id 가 그대로다 — test_sources_ids.py 가 확인.
ALLOWED_SOURCES_EDIT = ("self._new_id(", "self._id(")


def test_sources_registry_definitions_unchanged():
    cur = (ROOT / "tools/sources.py").read_text(encoding="utf-8").replace(*ALLOWED_SOURCES_EDIT)
    old, new = _defs(_at_tag("tools/sources.py")), _defs(cur)
    assert set(SOURCES_FROZEN) <= set(old)
    changed = [k for k in SOURCES_FROZEN if new.get(k) != old[k]]
    assert not changed, f"tools/sources.py 근거 등록 정의가 바뀜(추가만 허용): {changed}"


def test_config_existing_values_unchanged():
    old = _leaves(yaml.safe_load(_at_tag("config.yaml")))
    new = _leaves(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")))
    missing = [k for k in old if k not in new and not _removable(k)]
    changed = [k for k in old if k in new and new[k] != old[k]]
    assert not missing, f"config.yaml 기존 키 삭제: {missing}"
    assert not changed, f"config.yaml 기존 값 변경(추가만 허용): {changed}"
