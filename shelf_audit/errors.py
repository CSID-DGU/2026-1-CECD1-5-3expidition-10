# -*- coding: utf-8 -*-
"""
오류 전파.

v4 의 문제: 단계 안에서 `except Exception: pass` 로 삼키거나, 전체가 터져 500 하나로만 보고됐다.
v5 규칙:
  * 치명 오류(ShelfAuditError 계열)는 그대로 올라가 output 층에서 status="error" + code 로 변환된다.
  * 비치명 단계는 `with diag.stage("name", critical=False)` 로 감싼다. 예외가 나면 기록하고 계속 진행하되,
    해당 단계에 의존하는 판정은 Diagnostics.degraded 를 보고 자기 confidence 를 낮춘다.
  * 모든 경고는 코드(code)와 메시지로 남아 analysis.diagnostics 로 나간다. 대시보드의 '확인 필요' 근거가 된다.
"""
import contextlib
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class ShelfAuditError(Exception):
    code = "internal_error"

    def __init__(self, message: str, **info: Any):
        super().__init__(message)
        self.info = info


class InputError(ShelfAuditError):
    code = "bad_input"


class DetectionError(ShelfAuditError):
    code = "detection_failed"


class ReferenceError(ShelfAuditError):
    code = "reference_invalid"


class ShelfIdentificationError(ShelfAuditError):
    code = "shelf_not_identified"


@dataclass
class Diagnostics:
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    degraded: Set[str] = field(default_factory=set)     # 실패했거나 신뢰할 수 없는 단계 이름

    def warn(self, code: str, message: str, **info: Any) -> None:
        self.warnings.append({"code": code, "message": message, **info})

    def is_degraded(self, *names: str) -> bool:
        return any(n in self.degraded for n in names)

    @contextlib.contextmanager
    def stage(self, name: str, critical: bool = False):
        """단계 실행 기록. critical=False 인 단계의 예외는 기록 후 삼킨다(결과는 degraded 에 표시)."""
        t0 = time.time()
        rec = {"ok": True, "ms": 0}
        self.stages[name] = rec
        try:
            yield rec
        except ShelfAuditError:
            rec["ok"] = False
            self.degraded.add(name)
            raise
        except Exception as e:                   # noqa: BLE001
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc(limit=4)
            self.degraded.add(name)
            self.warn("stage_failed", f"{name} 단계 실패: {e}", stage=name)
            if critical:
                raise ShelfAuditError(f"{name} 단계 실패: {e}", stage=name) from e
        finally:
            rec["ms"] = int((time.time() - t0) * 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {"warnings": list(self.warnings),
                "degraded": sorted(self.degraded),
                "stages": {k: {kk: vv for kk, vv in v.items() if kk != "trace"} for k, v in self.stages.items()}}


def error_payload(name: str, err: Exception, diag: Optional[Diagnostics] = None) -> Dict[str, Any]:
    """예외 -> 백엔드 error 응답(형태는 v4 와 동일: status/filename/vision_items/message). code 는 message 안에 넣는다."""
    code = getattr(err, "code", "internal_error")
    out = {"status": "error", "filename": name, "vision_items": [], "message": f"[{code}] {err}"}
    if diag is not None:
        out["analysis"] = {"diagnostics": diag.to_dict()}
    return out
