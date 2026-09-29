"""P0 계약 커밋 검사: State(C1)·rubric(C5)·config(C6) 필드, 가격표 이전, 교차 함수 시그니처(C3·C4·C10), v1 호환 answer_question."""
from __future__ import annotations

import inspect
import re
import subprocess
import typing

import pytest
import yaml

from core.config import ROOT, get_config

C1_KEYS = ["domain", "run_date", "registry", "discovery_rounds", "raw_candidates", "seen", "screened", "queue",
           "current", "iterations", "founder", "tech", "market", "competition", "market_cache", "scorecard",
           "decision", "evaluations", "end_reason", "report", "rag_traces", "log"]
DISCOVERY_IN = {"domain", "run_date", "registry", "discovery_rounds", "seen", "queue", "iterations"}
DISCOVERY_OUT = {"registry", "discovery_rounds", "raw_candidates", "seen", "screened", "queue", "current", "iterations",
                 "founder", "tech", "market", "competition", "scorecard", "end_reason", "rag_traces", "log"}
REDUCERS = {"registry": "merge_dict", "seen": "add_unique", "screened": "add", "market_cache": "merge_dict",
            "evaluations": "add", "rag_traces": "add", "log": "add"}


# ── C1 State
def test_state_keys_and_reducers():
    from graph import state as st

    hints = typing.get_type_hints(st.InvestmentState, include_extras=True)
    assert list(hints) == C1_KEYS
    for k, fn in REDUCERS.items():
        assert hints[k].__metadata__[0].__name__ == fn, k
    assert set(typing.get_type_hints(st.DiscoveryIn)) == DISCOVERY_IN
    assert set(typing.get_type_hints(st.DiscoveryOut)) == DISCOVERY_OUT
    # 서브그래프 입력에 누적 키가 없어야 부모 Reducer 가 같은 기록을 두 번 붙이지 않는다
    assert not DISCOVERY_IN & {"screened", "log", "rag_traces", "evaluations"}


def test_state_inline_comments_parsed_by_design_doc():
    """설계서 State 표(docs/build_design._state_fields)와 같은 정규식으로 22키가 모두 읽혀야 한다."""
    src = (ROOT / "graph" / "state.py").read_text(encoding="utf-8")
    body = src[src.index("class InvestmentState"):]
    names = [m.group(1) for m in re.finditer(r"^ {4}(\w+): ([^#\n]+?)[ \t]*#[ \t]*([^\n]+)$", body, re.M)]
    assert names == C1_KEYS


def test_v1_graph_still_compiles_with_new_state():
    from graph.builder import build_graph

    assert build_graph() is not None


# ── C5 rubric
def test_rubric_contract_fields():
    from core.judge import load_rubric

    r = load_rubric()
    owners = {d["id"]: d["owner"] for d in r["dimensions"]}
    assert owners == {"founder": "founder", "market": "market", "product": "tech", "competition": "competition",
                      "traction": "decision", "deal": "decision"}
    qs = [q for d in r["dimensions"] for q in d["questions"]]
    assert len(qs) == 24
    for q in qs:
        assert q["point"] and q["dd_item"], q["id"]
        assert all(1 <= b <= 10 for b in q["bessemer_q"]), q["id"]
    assert [b["q"] for b in r["bessemer"]] == list(range(1, 11))
    related = {n: {q["id"] for q in qs if n in q["bessemer_q"]} for n in range(1, 11)}
    for b in r["bessemer"]:
        assert b["text"] and isinstance(b["proxy"], bool)
        assert set(b["via"]) <= related[b["q"]], f"Q{b['q']} via 는 bessemer_q 가 그 번호인 문항이어야 함"
    assert [b["q"] for b in r["bessemer"] if b["proxy"]] == [10]


def test_rubric_v1_fields_unchanged():
    """text·need·플래그·short·기존 bessemer 문구·가중치·관문·Deal-killer 는 v1 그대로."""
    out = subprocess.run(["git", "show", "v1-safe:rubric.yaml"], cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        pytest.skip("v1-safe 태그 없음")
    old = yaml.safe_load(out.stdout.decode("utf-8"))
    from core.judge import load_rubric

    new = load_rubric()
    for od, nd in zip(old["dimensions"], new["dimensions"], strict=True):
        assert {k: nd[k] for k in od if k != "questions"} == {k: v for k, v in od.items() if k != "questions"}
        for oq, nq in zip(od["questions"], nd["questions"], strict=True):
            assert {k: nq.get(k) for k in oq} == oq, oq["id"]
    assert new["gates"] == old["gates"] and new["deal_killers"] == old["deal_killers"]


# ── C6 config
def test_config_contract_keys():
    cfg = get_config()
    w, rag, d, roi, rep = cfg.workflow, cfg.rag, cfg.decision, cfg.roi, cfg.report
    assert (w.stop_on_invest, w.calibrate, w.recursion_limit) == (True, False, 100)
    assert (rag.max_rewrites, rag.max_regenerations, rag.rag_recursion_limit) == (2, 1, 25)
    assert rag.allow_direct_answer is False  # 분석 에이전트 호출은 direct 경로를 쓰지 않는다
    assert (cfg.tools.summarize_max_chars, cfg.tools.summarize_chunk_chars) == (6000, 3000)
    assert (d.method, d.threshold, d.step, d.clip, d.min_founder_yes) == ("payne_relative", 1.10, 0.5, [0.5, 1.5], 1)
    assert (d.info_gap_ratio, d.reference_file, d.reference_min_n) == (0.6, "data/reference_class.json", 5)
    assert (d.sensitivity, d.flip_max_items, d.dd_max_items) == ([1.00, 1.05, 1.10, 1.15, 1.20], 3, 5)
    assert (roi.fx_krw_per_usd, roi.stake_assumption, roi.target_multiple) == (1400, [0.10, 0.20], 10)
    assert roi.stage_median_usd_m == {"Seed": 1, "Pre-A": 1, "Series A": 7, "Pre-B": 7, "Series B": 14,
                                      "Pre-C": 14, "Series C": 14}
    assert "exit_share_warning" not in roi  # ROI 는 단계 중앙값 대비·VC Method 필요 Exit 만 계산
    assert (rep.scenario, rep.max_rewrites) == (False, 2)


# ── 가격표 이전 (models.price_per_mtok)
def test_price_table_from_config():
    from core.cost import model_price

    table = get_config().models.price_per_mtok
    for role in ("generator", "judge"):
        assert get_config().models[role] in table
    m = get_config().models.generator
    assert model_price(m) == tuple(table[m])
    assert model_price(m + "-2025-04-14") == tuple(table[m])        # 응답 모델명의 날짜 접미사
    assert model_price("unknown-model") == tuple(table[m])          # 표에 없으면 생성 모델 가격


def test_price_change_in_config_is_used(set_cfg):
    from core.cost import model_price

    m = get_config().models.generator
    set_cfg(("models", "price_per_mtok", m), [1.0, 2.0])
    assert model_price(m) == (1.0, 2.0)


def test_no_model_name_literal_in_cost_code():
    """모델명은 config.yaml 에만 둔다 (가격표를 옮긴 두 파일에 모델명이 남아 있지 않아야 함)."""
    for rel in ("core/cost.py", "eval/api_usage.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert not [m for m in get_config().models.price_per_mtok if m in text], rel


# ── C3 판정 공용 모듈
def test_load_rubric_reexported_from_decision():
    import agents.decision as dec
    import core.judge as judge

    assert dec.load_rubric is judge.load_rubric


def test_company_keys_and_evidence_pool():
    from core.judge import company_keys, evidence_pool

    c = {"name": "팜랩", "official_name": "(주)팜랩", "name_en": "Farm Lab", "evidence_ids": ["W1", "W2", "Wx"]}
    assert company_keys(c) == ["farmlab", "주팜랩", "팜랩"]
    state = {"current": c, "registry": {i: {"id": i} for i in ("W1", "W2", "W3", "D1", "D2")},
             "founder": {"pool_ids": ["W3", "W1"]}, "tech": {"pool_ids": ["D1"]}, "market": {}, "competition": {}}
    assert evidence_pool(state, extra=["D2", "D9"]) == ["W1", "W2", "W3", "D1", "D2"]


def test_judge_dimension_signature():
    from core.judge import judge_dimension
    from tools.sources import SourceRegistry

    assert list(inspect.signature(judge_dimension).parameters) == ["dim_id", "company", "pool_ids", "reg", "analysis",
                                                                   "run_date"]
    # 근거가 없으면 LLM 을 부르지 않고 네 문항 모두 미확인 (C3 모양)
    r = judge_dimension("founder", {"official_name": "x"}, [], SourceRegistry(), "", "2026-09-30")
    assert [row["answer"] for row in r["rows"]] == ["UNKNOWN"] * 4
    assert {"dim", "name", "weight", "owner", "rows", "yes", "no", "unknown", "n"} <= set(r)


# ── C10 투자 판단 공개 함수 (P3 가 구현)
@pytest.mark.parametrize("name, params", [
    ("load_reference", ["exclude"]),
    ("write_reference_class", ["evaluations", "path", "run_date"]),
    ("write_threshold_sensitivity", ["ref_path", "out_path"]),
    ("payne_multiplier", ["rows", "mean", "rubric", "step", "clip"]),
    ("decide_rule", ["M", "founder_yes", "killers", "unknown_ratio", "cfg"]),
    ("flip_conditions", ["rows", "mean", "rubric", "cfg", "founder_ok", "killers"]),
    ("bessemer_panel", ["rows", "rubric"]),
    ("roi", ["current", "market", "cfg"]),
    ("parse_amount", ["s"]),
])
def test_decision_public_signatures(name, params):
    import agents.decision as dec

    assert list(inspect.signature(getattr(dec, name)).parameters) == params


# ── C4 RAG·도구, 👤 창업자
def test_make_tools_and_founder_contract():
    from agents.founder import founder_node
    from tools.agent_tools import TOOL_NAMES, make_tools
    from tools.sources import SourceRegistry

    assert TOOL_NAMES == ("search_documents", "web_search", "summarize_document")  # fetch_page 없음
    assert tuple(make_tools(SourceRegistry(), "tech")) == TOOL_NAMES
    assert list(inspect.signature(founder_node).parameters) == ["state"]


def test_answer_question_v1_compat(monkeypatch):
    import rag.agentic_rag as ar
    from tools.sources import SourceRegistry

    sig = inspect.signature(ar.answer_question)
    assert sig.parameters["allow_direct"].default is False and sig.parameters["allow_web"].default is True
    trace = [{"query": "q", "retrieved": 8, "relevant": 1}, {"query": "q2", "retrieved": 8, "relevant": 1},
             {"query": "q", "web_fallback": 2}]

    def fake(question, purpose, registry, agent):
        registry.data["W1"] = {"id": "W1"}
        return ["D1", "W1"], trace

    monkeypatch.setattr(ar, "agentic_rag", fake)
    reg = SourceRegistry()
    out = ar.answer_question_v1("질문", "목적", reg, "market")  # v1 교정형 경로(생성 없음)는 폴백으로 남아 있다
    assert out == {"answer": "", "evidence_ids": ["D1", "W1"], "cited_ids": [], "status": "grounded", "route": "docs",
                   "rewrites": 1, "regenerations": 0, "trace": trace}
    assert "W1" in reg.data  # registry 는 제자리에서 갱신

    monkeypatch.setattr(ar, "agentic_rag", lambda *a: ([], [{"query": "q", "retrieved": 0, "relevant": 0}]))
    out = ar.answer_question_v1("질문", "목적", SourceRegistry(), "tech")
    assert (out["status"], out["rewrites"], out["evidence_ids"]) == ("not_found", 0, [])
