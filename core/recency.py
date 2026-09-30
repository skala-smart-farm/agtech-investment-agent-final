"""최근성 규칙 한 곳 ('최근 24개월'의 정의). 판정(core/judge.py)·탐색 관문 보정(graph/discovery_graph.py)이 함께 쓰고,
발굴 채널(tools/channels.py 와우테일 게시일 필터)도 같은 규칙을 따른다.

- 최근 N개월 = [평가 기준일 − N개월, 평가 기준일]. 평가 기준일(run_date)보다 뒤의 날짜는 '최근'이 아니라 아직 일어나지 않은
  일(예정·계획)이므로 최근 근거·최근 투자로 인정하지 않고 미확인으로 둔다.
- 날짜는 적힌 정밀도로만 비교한다. 'YYYY-MM' 은 달 단위(기준일과 같은 달이면 기준일 이전으로 본다),
  'YYYY' 는 연 단위(기준 연도와 같으면 이전으로 본다). 날짜를 읽지 못하면 판단하지 않는다.
- 설계서의 "사건 날짜가 평가일 기준 24개월 안인지는 LLM 에 맡기지 않고 날짜 규칙으로 다시 확인한다"를 미래 쪽까지 지킨다
  (원래 코드는 '24개월보다 오래됨'만 걸러, 기준일 이후 날짜가 음수 개월로 통과했다).
- 문장 규칙(최근 24개월 문항 F3·R1·R2 의 YES 인용): 인용 속 사건 날짜가 기준일 이후뿐이면(only_future_events),
  또는 완료 표지 없는 예정·계획·목표 문장이면(planned_only) 최근 사건 근거가 아니다. 시장 전망 문항(M1·M2)은 발행 시점
  기준의 전망이라 적용하지 않는다. 표현 목록은 서술어로 쓰인 것만 잡는다('출하 예정일'·'목표 온도' 같은 기능 이름 제외).
- 적격성 에이전트(agents/eligibility.py)는 캐시 보존 동결 파일이라, G2 의 미래·예정 라운드 규칙은 그 결과를 받는
  탐색 서브그래프 screen 단계와 D1 판정에서 이 모듈로 적용한다.
"""
from __future__ import annotations

import re

# 라운드 시점·마일스톤 날짜 필드: 'YYYY', 'YYYY-MM', 'YYYY.MM', 'YYYY년 M월'
_FIELD = re.compile(r"((?:19|20)\d{2})(?:\s*[-./년]\s*(\d{1,2}))?")
# 문장 속 사건 날짜. 달 표시가 분명한 것만 읽는다 ('2034년 1,172억 달러'의 1 을 1월로 읽지 않게)
_TEXT_YM = re.compile(r"((?:19|20)\d{2})\s*(?:년\s*(\d{1,2})\s*월|[.\-/]\s*(\d{1,2})(?![\d,]))")
_TEXT_Y = re.compile(r"((?:19|20)\d{2})\s*년(?!\s*\d|형)")   # 연도만: '2027년까지', '2027년 상반기' ('2027년형' 제품명 제외)
_TEXT_RANGE = re.compile(r"((?:19|20)\d{2})\s*[~\-–]\s*(?:19|20)\d{2}\s*년")   # '2024~2027년' 의 시작 연도
# 영어 기사(해외 후보): 'March 2027', 'Mar. 2027', 'by 2027', 'in 2027' (금액·모델명 속 숫자는 읽지 않게 앞말이 있는 것만)
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_EN_YM = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?"
                    r"|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?\s+(?:\d{1,2},?\s+)?((?:19|20)\d{2})\b", re.I)
_EN_Y = re.compile(r"\b(?:in|by|until|through|during|since|from|early|late|mid)[\s-]+((?:19|20)\d{2})\b(?![\d,.]\d)", re.I)

# 라운드 자체가 예정·추진인 문장 (투자금 사용 계획 '…해외 진출 계획'이나 'will use the funds'는 잡지 않게 라운드 표현에 붙은 것만)
ROUND_PLANNED = re.compile(
    r"(?:투자\s*유치|유치|라운드|시리즈\s*[a-z]?|펀딩)\s*(?:를|을)?\s*(?:추진|예정|계획|목표|준비|앞두)"
    r"|(?:추진|예정|계획|목표|준비)\s*(?:중인|하는|한)\s*(?:투자|라운드|시리즈)"
    r"|\b(?:plans?|aims?|expects?|looks?|looking|seeks?|seeking|in talks)\s+to\s+(?:raise|close)\b"
    r"|\bis\s+(?:currently\s+)?raising\s+(?:(?:a|an|its|new|funds|capital|money|series|seed)\b|\$)"
    r"|\bwill\s+(?:raise|close)\b", re.I)
# 같은 인용에 끝난 라운드가 함께 적혀 있으면('시드 투자를 유치했으며 내년 시리즈A 를 준비') 예정 라운드로 보지 않는다
ROUND_DONE = re.compile(r"유치했|유치하며|유치한\s|유치에\s*성공|유치\s*[…,.·!]|유치\s*$|받았|확보했|마무리|마쳤"
                        r"|\b(?:raised|closed|secured|completed|landed|bagged)\b", re.I)

# 완료되지 않은 일(예정·계획·목표) 표현. 서술어로 쓰인 것만 잡는다
# ('출하 예정일 예측', '재배 계획 수립', '목표 온도 제어' 같은 기능 이름을 계획 문장으로 읽지 않게)
_RIEUL = "[" + "".join(chr(0xAC00 + i * 28 + 8) for i in range(19 * 21)) + "]"   # 받침 ㄹ 글자(할·될·늘릴…)
PLANNED = re.compile(
    rf"{_RIEUL}\s*(?:계획|예정|방침|전망)"                       # 설치할 계획, 출시될 예정, 늘릴 방침
    r"|(?:예정|계획)(?:이다|이며|이고|이라|입니다|이었|으로|인\s)|목표(?:다|이다|이며|이고|로|입니다)"
    r"|(?:예정|계획|목표)\s*(?:[.,。)\]…]|$)"                    # 제목·문장 끝 '출시 예정'
    r"|것으로\s*(?:예상|전망|기대)|앞두고|앞둔|내년|\bnext\s+year\b"
    r"|\b(?:plans?|aims?|expects?|intends?|hopes?)\s+to\b|\bis\s+(?:set|scheduled|expected|poised)\s+to\b", re.I)
# 완료 표지. 계획 표현과 함께 있으면 완료된 사실이 함께 적힌 문장으로 본다 ('하기로 했다'·'추진하고 있다'는 완료가 아님)
_NOT_YET = r"(?!\s*(?:할|될|하기|예정|목표|계획))"
DONE = re.compile(
    r"(?<!기로 )(?<!기로)(?:했|하였|됐|되었)(?:다|으며|고|습니다)"
    r"|(?<!추진)(?<!계획)(?<!준비)(?<!검토)(?<!모색)(?<!협의)(?<!논의)하고\s*있(?:다|으며|습니다)"
    rf"|(?:운영|사용|판매|공급|가동)\s*중|(?:완료|달성|기록|체결|수상){_NOT_YET}|선정됐|선정되었|유치했|받았"
    r"|\b(?:raised|closed|secured|deployed|installed|launched|operates|operating|serves|served|signed|won)\b", re.I)


def parse_date(when) -> tuple[int, int | None] | None:
    """날짜 필드 → (연, 월 | None). 연도를 못 읽으면 None, 월이 1~12 밖이면 월 None."""
    m = _FIELD.match(str(when or "").strip())
    if not m:
        return None
    mo = int(m.group(2)) if m.group(2) else None
    return int(m.group(1)), (mo if mo and 1 <= mo <= 12 else None)


def _after(y: int, mo: int | None, run_date: str) -> bool:
    ry, rm = int(run_date[:4]), int(run_date[5:7])
    return y > ry if mo is None else (y, mo) > (ry, rm)


def is_future(when, run_date: str) -> bool:
    """when 이 평가 기준일보다 뒤인가 (적힌 정밀도로 비교). 날짜를 못 읽으면 False(판단하지 않음)."""
    p = parse_date(when)
    return p is not None and _after(*p, run_date)


def in_window(y: int, mo: int | None, run_date: str, months: int = 24) -> bool:
    """(연, 월 | None) 이 [평가 기준일 − months 개월, 평가 기준일] 안인가. 연도만 있으면 기준 연도 − months/12 ~ 기준 연도."""
    if _after(y, mo, run_date):
        return False
    ry, rm = int(run_date[:4]), int(run_date[5:7])
    return y >= ry - months // 12 if mo is None else (ry - y) * 12 + (rm - mo) <= months


def text_dates(text: str) -> list[tuple[int, int | None]]:
    """문장 속 사건 날짜 [(연, 월 | None)]. 달이 있는 표기('2027년 3월'·'2027.03'·'March 2027')와
    연도만 적은 표기('2027년까지'·'by 2027')를 읽는다."""
    text = text or ""
    out = []
    for m in _TEXT_YM.finditer(text):
        mo = int(m.group(2) or m.group(3))
        if 1 <= mo <= 12:
            out.append((int(m.group(1)), mo))
    out += [(int(y), None) for y in _TEXT_Y.findall(text) + _TEXT_RANGE.findall(text)]
    out += [(int(y), _MONTHS.index(mon[:3].lower()) + 1) for mon, y in _EN_YM.findall(text)]
    ym = {m.start(2) for m in _EN_YM.finditer(text)}   # 'in March 2027' 의 연도를 연도만 적은 표기로 또 세지 않게
    out += [(int(m.group(1)), None) for m in _EN_Y.finditer(text) if m.start(1) not in ym]
    return out


def only_future_events(text: str, run_date: str, months: int = 24) -> bool:
    """문장 속 사건 날짜 중 창(최근 months 개월) 안의 날짜는 없고 기준일 이후 날짜가 있으면 True (최근 사건 근거가 아님).
    다만 날짜 없는 완료 사실이 함께 적혀 있고('12개 농가에 설치했고 2027년까지 500곳으로') 오래된 날짜가 없으면 False:
    그 완료 사실의 최근성은 다른 인용처럼 근거 게시일로 본다. (전부 오래된 경우는 core/judge.py _events_too_old 가 거른다.)"""
    dates = text_dates(text)
    future = [d for d in dates if _after(*d, run_date)]
    if not future or any(in_window(*d, run_date, months) for d in dates):
        return False
    return len(future) < len(dates) or not DONE.search(text or "")


def round_planned(quote: str) -> bool:
    """투자 단계 인용이 끝난 라운드가 아니라 예정·추진 중인 라운드'만' 말하는가 (끝난 라운드가 함께 적혀 있으면 False)."""
    return bool(ROUND_PLANNED.search(quote or "")) and not ROUND_DONE.search(quote or "")


def future_round(record: dict, run_date: str) -> str | None:
    """적격성 기록의 최근 라운드가 기준일 이후이거나 예정 라운드면 그 이유(G2 불인정), 아니면 None."""
    rd = str(record.get("round_date") or "")
    if is_future(rd, run_date):
        return f"최근 라운드 시점({rd})이 평가 기준일({run_date})보다 뒤 — 완료된 투자로 확인되지 않음"
    if round_planned(record.get("stage_quote") or ""):
        return "단계 인용이 예정·추진 중인 라운드 — 완료된 투자로 확인되지 않음"
    return None


def planned_only(text: str, run_date: str, months: int = 24) -> bool:
    """예정·계획·목표 문장인데 완료 표지도, 창(최근 months 개월) 안의 사건 날짜도 없으면 True (완료된 사건이 아님)."""
    return (bool(PLANNED.search(text or "")) and not DONE.search(text or "")
            and not any(in_window(*d, run_date, months) for d in text_dates(text)))
