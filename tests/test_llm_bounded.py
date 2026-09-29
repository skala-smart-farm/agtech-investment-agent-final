"""v2 새 호출용 LLM: 출력 길이 한도(반복 퇴행)에 닿으면 반복 금지 문구를 붙여 한 번만 다시 묻는다."""
from __future__ import annotations

import pytest
from openai import LengthFinishReasonError
from pydantic import BaseModel

import core.llm_bounded as lb


class Out(BaseModel):
    x: str


def _length_error() -> LengthFinishReasonError:
    return LengthFinishReasonError.__new__(LengthFinishReasonError)


def test_retry_once_with_hint_then_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(lb, "_registry_path", lambda: tmp_path / "degenerate.json")
    prompts, replies = [], [_length_error(), Out(x="ok")]

    class Fake:
        def invoke(self, prompt):
            prompts.append(prompt)
            r = replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

    penalties = []
    monkeypatch.setattr(lb.Bounded, "_llm", lambda self, penalty=0.0: penalties.append(penalty) or Fake())
    assert lb.bounded(Out).invoke("질문") == Out(x="ok")
    assert prompts == ["질문", "질문" + lb.REPEAT_HINT] and penalties == [0.0, lb.REPEAT_PENALTY]

    # 퇴행이 기록된 프롬프트는 다음부터 첫 시도 없이 바로 재시도 설정 (재현·비용)
    prompts.clear(); penalties.clear(); replies[:] = [Out(x="again")]
    assert lb.bounded(Out).invoke("질문") == Out(x="again") and penalties == [lb.REPEAT_PENALTY]

    replies[:] = [_length_error(), _length_error()]
    with pytest.raises(LengthFinishReasonError):  # 두 번째도 한도면 그대로 실패 (무한 재시도 없음)
        lb.bounded(Out).invoke("다른 질문")


def test_offline_miss_falls_back_to_cached_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(lb, "_registry_path", lambda: tmp_path / "degenerate.json")
    calls = []

    class Fake:
        def __init__(self, penalty):
            self.penalty = penalty

        def invoke(self, prompt):
            calls.append(self.penalty)
            if not self.penalty:
                raise RuntimeError("--offline: 재현용 캐시에 없는 LLM 호출입니다")
            return Out(x="cached")

    monkeypatch.setattr(lb.Bounded, "_llm", lambda self, penalty=0.0: Fake(penalty))
    assert lb.bounded(Out).invoke("q") == Out(x="cached") and calls == [0.0, lb.REPEAT_PENALTY]
    assert lb._key(Out, "q") in lb._known()


def test_caps_from_config():
    from core.config import get_config

    caps = get_config().models.max_output_tokens
    assert caps["default"] >= 944 and caps["report"] >= 1947  # v1 실측 최대 출력보다 커야 정상 응답을 자르지 않는다
