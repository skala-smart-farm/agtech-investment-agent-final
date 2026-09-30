"""구조화된 발굴 채널 (키 불필요, 날짜가 정확한 공개 데이터).

- TIPS 창업기업 공개 목록: 대표자·설립일·선정연도·운영사(=선투자자)가 필드로 있다.
  TIPS 는 운영사가 먼저 투자해야 선정되므로, 선정 자체가 "Seed 급 투자를 받은 회사"라는 신호다.
- 와우테일 '애그테크' 카테고리 (WordPress REST): 투자 유치 기사의 정확한 게시일을 준다.
둘 다 날짜별 스냅샷을 캐시에 저장해 같은 날 다시 실행하면 같은 입력을 쓴다(재현성).
수집 대상 사이트의 robots·차단 정책을 우회하지 않으며, 요청 사이에 간격을 둔다.
"""
from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime
from functools import lru_cache

import requests

from core.config import get_config, path
from tools.listing_check import normalize

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/128.0 Safari/537.36"}
AGRI = re.compile(r"농업|농산|농작물|작물|재배|스마트팜|스마트 팜|수직농장|식물공장|병해충|병충해|축산|가축|양돈|양계|낙농|"
                  r"한우|수확|농기계|농장|과수|온실|시설원예|육묘|애그|agri|agtech|farm|crop|livestock", re.I)
AI = re.compile(r"AI|인공지능|머신러닝|딥러닝|컴퓨터 ?비전|로봇|자율주행|드론|센서|IoT|데이터", re.I)
FUNDING = re.compile(r"투자\s?유치|시드|프리\s?A|시리즈|브릿지|팁스|TIPS", re.I)


def _snapshot(name: str):
    """같은 캐시 폴더에 이전 스냅샷이 있으면 그것을 쓴다 (제출본 재현). 없으면 오늘 날짜로 새로 받는다."""
    d = path(f"{get_config().cache.dir}/snapshots/.keep").parent
    old = sorted(d.glob(f"{name}_*.json"))
    return old[-1] if old else d / f"{name}_{datetime.now():%Y%m%d}.json"


def tips_agtech(min_year: int = 2021) -> list[dict]:
    """TIPS 선정 기업 중 농업 분야이면서 AI·로봇·센서 신호가 있는 기업."""
    f = _snapshot("tips")
    if f.exists():
        rows = json.loads(f.read_text(encoding="utf-8"))
    else:
        api, rows, page = "https://jointips.or.kr/api/cms/public/startup/list", [], 1
        try:
            while True:
                j = requests.get(api, params={"page": page, "size": 100, "sort": "registered", "dir": "asc"},
                                 headers=UA, timeout=30).json()
                rows += j.get("data") or []
                if not j.get("data") or len(rows) >= j.get("total", 0):
                    break
                page += 1
                time.sleep(0.3)
        except Exception as e:  # 채널 하나가 실패해도 다른 채널로 계속 진행
            print(f"   (TIPS 채널 실패: {e})")
            return []
        f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    out = []
    for r in rows:
        intro, name = r.get("intro") or "", r.get("name") or ""
        year = int(str(r.get("selYear") or 0)[:4] or 0)
        if year >= min_year and (AGRI.search(intro) or AGRI.search(name)) and AI.search(intro):
            out.append({"name": name, "ceo": r.get("ceo") or "", "intro": intro[:300], "founded": r.get("estDt"),
                        "sel_year": year, "operator": r.get("operatorName"), "homepage": r.get("homepageUrl") or ""})
    return out


@lru_cache(maxsize=2)
def _tips_index(snapshot) -> dict[str, list[dict]]:
    """정규화한 회사명 → 농업 분야 TIPS 기업 행 (선정연도 제한 없음)."""
    idx: dict[str, list[dict]] = {}
    for r in json.loads(snapshot.read_text(encoding="utf-8")):
        name = r.get("name") or ""
        if AGRI.search(r.get("intro") or "") or AGRI.search(name):
            idx.setdefault(normalize(name), []).append(
                {"name": name, "ceo": r.get("ceo") or "", "founded": r.get("estDt") or "", "sel_year": r.get("selYear")})
    return idx


def tips_profile(names: list[str]) -> dict | None:
    """TIPS 목록에서 이름이 같은 농업 분야 기업의 대표자·설립일(estDt, YYYY-MM-DD).
    발굴 단계가 받아 둔 스냅샷만 쓰고 새로 받지 않는다. 스냅샷이 없거나 동명 기업이 둘 이상이면 None."""
    f = _snapshot("tips")
    if not f.exists():
        return None
    idx = _tips_index(f)
    for n in names:
        hits = idx.get(normalize(n)) if n else None
        if hits and len({(h["ceo"], h["founded"]) for h in hits}) == 1:
            return hits[0]
    return None


def wowtale_agtech_funding(since_days: int = 730) -> list[dict]:
    """와우테일 애그테크 카테고리의 투자 유치 기사 (최근 2년)."""
    f = _snapshot("wowtale_agtech")
    if f.exists():
        posts = json.loads(f.read_text(encoding="utf-8"))
    else:
        posts, page = [], 1
        try:
            while True:
                r = requests.get("https://wowtale.net/wp-json/wp/v2/posts",
                                 params={"categories": 19425, "per_page": 100, "page": page,
                                         "_fields": "date,link,title"}, headers=UA, timeout=30)
                if r.status_code != 200:
                    break
                posts += r.json()
                if page >= int(r.headers.get("X-WP-TotalPages", 1)):
                    break
                page += 1
                time.sleep(2)
        except Exception as e:
            print(f"   (와우테일 채널 실패: {e})")
            return []
        f.write_text(json.dumps(posts, ensure_ascii=False), encoding="utf-8")
    from core.config import run_date

    out, now = [], datetime.strptime(run_date(), "%Y-%m-%d")
    for p in posts:
        title = html.unescape(p["title"]["rendered"])
        d = p["date"][:10]
        # 평가 기준일 이후 글(음수 일수)은 최근 2년에 넣지 않는다 (core/recency.py 와 같은 규칙)
        if FUNDING.search(title) and 0 <= (now - datetime.fromisoformat(d)).days <= since_days:
            out.append({"title": title, "date": d, "url": p["link"]})
    return out
