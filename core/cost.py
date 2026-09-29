"""LLM 토큰 사용량·비용 집계. LangSmith 없이도 실행마다 비용을 기록한다.

캐시(SQLite)에서 꺼낸 응답은 실제 API 호출이 아니므로 비용에 넣지 않는다(llm_output 이 없으면 캐시 응답).
"""
from __future__ import annotations

import threading

from langchain_core.callbacks import BaseCallbackHandler

from core.config import get_config


def model_price(model: str) -> tuple[float, float]:
    """모델 이름 → (입력, 출력) USD / 100만 토큰. 가격표는 config.yaml 의 models.price_per_mtok 에만 둔다.
    응답의 모델명에 날짜가 붙어도(…-2025-04-14) 가장 길게 일치하는 앞부분으로 찾고, 표에 없으면 생성 모델 가격으로 계산한다.
    프롬프트 캐시로 처리된 입력은 호출하는 쪽에서 입력 가격의 1/4 로 계산한다."""
    models = get_config().models
    table = models.get("price_per_mtok") or {}
    hit = max((k for k in table if model.startswith(k)), key=len, default=None)
    pi, po = table[hit] if hit else table[models.generator]
    return float(pi), float(po)


class CostTracker(BaseCallbackHandler):
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.cached = self.input_tokens = self.output_tokens = 0
        self.usd = 0.0

    def on_llm_end(self, response, **kwargs) -> None:
        out = response.llm_output or {}
        usage = out.get("token_usage") or {}
        with self.lock:
            if not out:
                self.cached += 1
                return
            model = str(out.get("model_name", ""))
            pi, po = model_price(model)
            i, o = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            ci = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            self.calls += 1
            self.input_tokens += i
            self.output_tokens += o
            self.usd += (i - ci) / 1e6 * pi + ci / 1e6 * pi / 4 + o / 1e6 * po

    def summary(self) -> dict:
        return {"api_calls": self.calls, "cache_hits": self.cached, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.usd, 4)}


TRACKER = CostTracker()
