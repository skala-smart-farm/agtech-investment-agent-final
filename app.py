"""AgTech 스타트업 투자 평가 에이전트 실행 스크립트.

    uv run python app.py                  # 탐색 → 창업자·기술·시장성·경쟁사 → 투자 판단 → 보고서(PDF)
    uv run python app.py --offline        # 재현 테스트: 저장소의 캐시(replay/)만 쓰고 API 키 없이 실행
    uv run python app.py --calibrate      # 보정 실행: 투자여도 멈추지 않고 평가 상한까지 평가해 동종 기준 집단
                                          #   (data/reference_class.json)과 기준 배수 민감도를 만든다. 보고서는 만들지 않는다
    uv run python app.py --threshold 1.30 --out outputs/scenario_hold   # 시나리오 실행: 기준 배수만 바꿔 다른 보고서 모드 확인
    uv run python app.py --graph-only     # 그래프 그림(docs/architecture.png, docs/architecture_langgraph.png)만 생성
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")  # 라이브러리 경고로 진행 로그가 묻히지 않게

from core.config import ROOT, get_config, path, require_keys
from core.config import run_date as get_run_date
from core.cost import TRACKER
from graph.builder import build_graph
from tools import web_search as web_search_mod

CALIBRATION_DIR = "outputs/calibration"                          # 보정 실행의 실행 기록 폴더
SENSITIVITY_FILE = "outputs/eval/threshold_sensitivity.json"     # 기준 배수별 투자 추천 수
SCREENED_KEYS = ("name", "region", "stage", "round_date", "founded_date", "ceo", "eligible", "reason",
                 "gate_search_failed", "channels")
EVALUATION_KEYS = ("name", "region", "stage", "decision", "hold_type", "multiplier", "score100", "reasons", "flip")
ROW_KEYS = ("qid", "answer", "rationale", "quote", "evidence_ids")


def save_graph_image(app) -> str:
    """README Architecture 그림 2종.
    - docs/architecture.png          : 설계서와 같은 한글 설명 그림 (docs.build_design._main_mermaid)
    - docs/architecture_langgraph.png: 컴파일된 LangGraph 가 직접 그린 그림. xray=1 이라 🔍 탐색 서브그래프 안쪽까지 그린다
                                       (코드와 설계가 같은지 확인용)"""
    from docs.build_design import _main_mermaid
    from report.mermaid import mermaid_to_png

    mmd = app.get_graph(xray=1).draw_mermaid()
    path("docs/architecture_langgraph.mmd").write_text(mmd, encoding="utf-8")
    mermaid_to_png(mmd, path("docs/architecture_langgraph.png"))
    out = path("docs/architecture.png")
    mermaid_to_png(_main_mermaid(get_config()), out)
    return str(out)


def _check_browser() -> None:
    """PDF 는 마지막 단계에서 만들기 때문에, Chromium 이 없으면 처음부터 알려 준다."""
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        if not Path(p.chromium.executable_path).exists():
            raise SystemExit("PDF 생성용 브라우저가 없습니다. 먼저 `uv run playwright install chromium` 을 실행하세요.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="저장소에 포함된 재현용 캐시(replay/) 대신 웹·LLM 을 새로 호출해 최신 정보로 평가")
    parser.add_argument("--offline", action="store_true",
                        help="재현 테스트: 캐시에 없는 검색·LLM 호출이 생기면 바로 실패 (API 키 없이 실행 가능)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="재현용 캐시에 실패로 표시된 검색만 다시 시도 (검색 한도 초과로 비었던 근거를 새 키로 보강)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="투자 기준 배수(decision.threshold)를 이번 실행에서만 바꾼다. 제출 보고서를 덮어쓰지 않게 --out 과 함께 쓴다")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--calibrate", action="store_true",
                      help=f"보정 실행: 투자여도 멈추지 않고 평가 상한까지 평가 → 기준 집단 파일 생성, 실행 기록은 {CALIBRATION_DIR}/")
    mode.add_argument("--out", metavar="DIR", default=None,
                      help="시나리오 실행: 결과를 DIR 에 쓰고 제출용 파일(RAG-Output_…)은 만들지 않는다")
    return parser.parse_args(argv)


def apply_options(args: argparse.Namespace, cfg) -> str:
    """실행 옵션을 설정에 반영하고 실행 모드(main | calibrate | scenario)를 돌려준다.
    get_config() 캐시 객체의 원본 dict 를 고치므로 이후 노드가 get_config() 로 읽어도 바뀐 값이 보인다."""
    if args.fresh:
        cfg["cache"]["dir"] = "cache_fresh"
    if args.threshold is not None:
        cfg["decision"]["threshold"] = args.threshold
    if args.calibrate:
        cfg["workflow"]["stop_on_invest"] = False
        cfg["workflow"]["calibrate"] = True
        cfg["report"]["output_dir"] = CALIBRATION_DIR
        return "calibrate"
    if args.out:
        cfg["report"]["output_dir"] = args.out
        cfg["report"]["scenario"] = True
        return "scenario"
    return "main"


def search_notice(args: argparse.Namespace, cfg) -> str:
    """검색 공급자 안내 한 줄. 재현 실행은 캐시만 읽으므로 키·공급자와 무관하다는 것을 먼저 알린다."""
    names = " → ".join(p.capitalize() for p in web_search_mod.active_providers()) or "없음"
    if not (args.fresh or args.retry_failed):
        return f"   검색: 재현용 캐시({cfg.cache.dir}/)에서 읽음 — 검색 키·공급자와 무관 (공급자 {names})"
    msg = f"   검색 공급자: {names} · 요청 제한 {cfg.search.get('timeout_sec', 30)}초"
    if web_search_mod.active_providers() == ["tavily"]:
        msg += (" — SERPER_API_KEY 가 없어 모든 검색을 Tavily 로 보냅니다. 새 평가는 검색이 수백 번이라 오래 걸리고"
                " Tavily 무료 한도를 많이 씁니다 (README '실행이 느리거나 멈춘 것 같을 때')")
    return msg


def initial_state(cfg, run_date: str) -> dict:
    return {"domain": cfg.domain.name, "run_date": run_date, "registry": {}, "discovery_rounds": 0, "iterations": 0,
            "queue": [], "seen": [], "evaluations": [], "screened": [], "log": [], "rag_traces": []}


def _evaluation_log(e: dict) -> dict:
    sc = e.get("scorecard") or {}
    return {**{k: e.get(k) for k in EVALUATION_KEYS},
            "roi": sc.get("roi"), "criteria": e.get("criteria", sc.get("criteria")),
            "rows": [{k: r.get(k) for k in ROW_KEYS} for r in sc.get("rows", [])]}


def build_run_log(state: dict, cfg, *, mode: str, run_date: str, elapsed: float) -> dict:
    return {
        "run_date": run_date, "elapsed_sec": elapsed, "mode": mode, "threshold": cfg.decision.threshold,
        "end_reason": state.get("end_reason"),
        "models": dict(cfg.models), "embedding": cfg.embedding.model,
        "discovery_rounds": state.get("discovery_rounds"), "evaluated": state.get("iterations"),
        "screened": [{k: r.get(k) for k in SCREENED_KEYS} for r in state.get("screened", [])],
        "evaluations": [_evaluation_log(e) for e in state.get("evaluations", [])],
        "report": state.get("report") or {}, "rag_traces": state.get("rag_traces", []), "log": state.get("log", []),
        "sources_collected": len(state.get("registry", {})),
        "failed_searches": sorted({(f["agent"], f["query"]) for f in getattr(web_search_mod, "FAILED_QUERIES", [])}),
        "llm_cost": TRACKER.summary(),
    }


def _write_json(rel: str, data: dict) -> None:
    # 공개 저장소에 로컬 절대경로가 남지 않게 저장소 기준 상대경로로 적는다
    text = json.dumps(data, ensure_ascii=False, indent=2).replace(str(ROOT) + "/", "")
    path(rel).write_text(text, encoding="utf-8")


def v1_evaluated_names() -> list[str] | None:
    """v1-safe 태그의 제출 실행 기록(outputs/run_log.json)에서 심층 평가한 후보 이름. 태그나 git 이 없으면 None."""
    try:
        out = subprocess.run(["git", "show", "v1-safe:outputs/run_log.json"], cwd=ROOT, capture_output=True)
    except OSError:
        return None
    if out.returncode != 0:
        return None
    return [e["name"] for e in json.loads(out.stdout.decode("utf-8")).get("evaluations", [])]


def check_v1_members(names: list[str]) -> bool | None:
    """캐시 보존 게이트: 발굴·관문을 v1 캐시 그대로 재생했다면 보정 실행이 평가한 후보는 v1 평가 후보와 같다."""
    v1 = v1_evaluated_names()
    if v1 is None:
        print("[캐시 보존 확인] v1-safe 태그를 찾지 못해 비교를 건너뜁니다.")
        return None
    added, missing = sorted(set(names) - set(v1)), sorted(set(v1) - set(names))
    if not added and not missing:
        print(f"[캐시 보존 확인] 평가 후보 {len(set(names))}곳이 v1 평가 후보와 같습니다.")
        return True
    print(f"[경고] 평가 후보가 v1 과 다릅니다 (추가 {added}, 빠짐 {missing}). 발굴·관문 캐시가 바뀌었는지 확인하세요.")
    return False


def finish_calibration(state: dict, cfg, run_date: str) -> dict:
    """보정 실행 뒤: 평가 결과로 동종 기준 집단 파일과 기준 배수 민감도를 쓰고 요약을 출력한다."""
    from agents.decision import write_reference_class, write_threshold_sensitivity

    evaluations = state.get("evaluations", [])
    ref_file = cfg.decision.reference_file
    ref = write_reference_class(evaluations, str(path(ref_file)), run_date)
    sens = write_threshold_sensitivity(str(path(ref_file)), str(path(SENSITIVITY_FILE)))
    n, need = ref.get("n") or 0, cfg.decision.reference_min_n
    enough = n >= need
    print(f"기준 집단: {n}곳 → {ref_file}")
    if not enough:  # 본 실행은 이 파일 대신 동종 평균 신호 0(fallback)으로 계산하게 된다
        print(f"[경고] 기준 집단 {n}곳 < 최소 {need}곳(decision.reference_min_n). 본 실행은 fallback(동종 평균 신호 0)으로 계산됩니다.")
    print(f"기준 배수별 투자 추천 수: {json.dumps(sens.get('invest_count_at', {}), ensure_ascii=False)} → {SENSITIVITY_FILE}")
    same = check_v1_members([m["name"] for m in ref.get("members") or []])  # 실제로 파일에 쓴 구성원을 v1 과 비교
    return {"reference_file": ref_file, "reference_n": n, "reference_enough": enough, "sensitivity_file": SENSITIVITY_FILE,
            "invest_count_at": sens.get("invest_count_at"), "same_as_v1_evaluated": same}


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # 파일로 내보낼 때도 진행 로그가 바로 보이게
    args = parse_args(argv)
    if args.retry_failed:
        os.environ["SEARCH_RETRY_FAILED"] = "1"
    if args.offline:
        os.environ["REPLAY_OFFLINE"] = "1"
    _check_browser()

    cfg = get_config()
    base_out = cfg.report.output_dir  # 비용 기록은 모드와 관계없이 기본 출력 폴더 한 곳에 모은다
    mode = apply_options(args, cfg)
    if args.fresh or args.retry_failed:  # 새로 호출할 실행은 키가 없으면 중간이 아니라 시작할 때 멈춘다
        require_keys()
    app = build_graph()
    if args.graph_only:
        print(save_graph_image(app))
        return

    t0 = time.time()
    run_date = get_run_date()
    meta = path(f"{cfg.cache.dir}/run_meta.json")
    if not meta.exists():  # 처음 실행한 날을 기준일로 기록 → 같은 캐시로 다시 돌리면 같은 날짜 기준으로 판정
        meta.write_text(json.dumps({"run_date": run_date}), encoding="utf-8")
    print(f"== AgTech 스타트업 투자 평가 시작 ({run_date}, 모드 {mode}, 기준 배수 {cfg.decision.threshold}, "
          f"생성 {cfg.models.generator}, 판정 {cfg.models.judge}, 임베딩 {cfg.embedding.model}, 캐시 {cfg.cache.dir}/) ==")
    print(search_notice(args, cfg))
    state = app.invoke(initial_state(cfg, run_date), {"recursion_limit": cfg.workflow.recursion_limit})
    elapsed = round(time.time() - t0, 1)

    out_dir = cfg.report.output_dir
    run_log = build_run_log(state, cfg, mode=mode, run_date=run_date, elapsed=elapsed)
    _write_json(f"{out_dir}/run_log.json", run_log)  # 보정 파일을 쓰다 실패해도 평가 기록은 남도록 먼저 저장
    cost = TRACKER.summary()
    if cost.get("api_calls"):  # 실제로 API 를 부른 실행만 비용 기록을 남긴다 (캐시 재생 실행은 0원이라 남기지 않음)
        with open(path(f"{base_out}/cost_history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_date": run_date, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "elapsed_sec": elapsed, "fresh": args.fresh, "mode": mode, **cost},
                               ensure_ascii=False) + "\n")
    print(f"== 완료 ({elapsed}초, 종료 사유 {state.get('end_reason')}, LLM {cost}) ==")

    report = state.get("report") or {}
    if mode == "calibrate":
        run_log["calibration"] = finish_calibration(state, cfg, run_date)
        _write_json(f"{out_dir}/run_log.json", run_log)
    elif report.get("submit_pdf") or report.get("pdf"):
        print(f"보고서: {report.get('submit_pdf') or report.get('pdf')}")
        if report.get("checks"):
            print(f"검증: {json.dumps(report['checks'], ensure_ascii=False)}")
    else:
        print(f"보고서를 만들지 못했습니다. {out_dir}/run_log.json 을 확인하세요.")


if __name__ == "__main__":
    main()
