# -*- coding: utf-8 -*-
"""
다중 상태: 한 책이 '기울어진 채 종이면을 보임'처럼 이상이 겹칠 수 있다.
판정 dict 에 states(겹친 이상 상태 전부, 우선순위 순)를 두고, state 는 그 첫째(백엔드 호환용 대표 상태)로 둔다.
normal / unknown / absent 는 states 가 비어 있다.
"""
from typing import Any, Dict, List, Optional

PRIORITY = ("abnormal_stack", "abnormal_paper", "abnormal_upside", "abnormal_tilted", "abnormal_depth")
# 함께 설 수 없는 짝: 누움(stack)은 다른 이상과 같이 내지 않는다 (누운 책의 기울기·뒤집힘·앞뒤 정렬은 뜻이 없다)
EXCLUSIVE = {"abnormal_stack"}


def ordered(states) -> List[str]:
    seen = []
    for s in PRIORITY:
        if s in states and s not in seen:
            seen.append(s)
    for s in states:                                     # 모르는 이름도 버리지 않는다
        if s not in seen and str(s).startswith("abnormal_"):
            seen.append(s)
    if any(s in EXCLUSIVE for s in seen):
        seen = [s for s in seen if s in EXCLUSIVE][:1]
    return seen


def normalize(r: Dict[str, Any]) -> Dict[str, Any]:
    """state 만 있는 옛 형식의 판정을 states 와 맞춘다."""
    st = r.get("state", "unknown")
    if "states" not in r or r["states"] is None:
        r["states"] = [st] if str(st).startswith("abnormal_") else []
    r["states"] = ordered(r["states"])
    if r["states"]:
        r["state"] = r["states"][0]
    elif str(st).startswith("abnormal_"):
        r["state"] = "normal"
    return r


def set_state(r: Dict[str, Any], state: str, conf: Optional[float] = None) -> None:
    """상태 하나로 바꾼다 (정상/판정불가/없음 또는 단일 이상)."""
    r["states"] = [state] if state.startswith("abnormal_") else []
    r["state"] = state
    if conf is not None:
        r["conf"] = conf


def add_state(r: Dict[str, Any], state: str, conf: Optional[float] = None) -> None:
    r["states"] = ordered(list(r.get("states") or []) + [state])
    r["state"] = r["states"][0]
    if conf is not None:
        r["conf"] = max(float(r.get("conf", 0.0)), conf)


def drop_state(r: Dict[str, Any], state: str, fallback: str = "normal", conf: Optional[float] = None) -> None:
    """겹친 이상 중 하나만 거둔다. 남는 게 없으면 fallback(정상)으로."""
    r["states"] = [s for s in (r.get("states") or []) if s != state]
    r["state"] = r["states"][0] if r["states"] else fallback
    if conf is not None and not r["states"]:
        r["conf"] = conf
