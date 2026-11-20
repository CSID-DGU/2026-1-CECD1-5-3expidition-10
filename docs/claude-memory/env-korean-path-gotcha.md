---
name: env-korean-path-gotcha
description: Windows 사용자 폴더 이름이 한글인 PC 에서는 Git Bash 에 바탕화면 같은 경로를 넘기면 깨진다 — PowerShell 이나 ctypes 로 경로를 얻을 것
metadata:
  type: feedback
---

Windows 사용자 프로필 폴더 이름이 한글인 PC 에서는, Bash 툴(Git Bash)로 사용자 폴더 아래 경로(바탕화면 등)를 cp/python 인자로 넘기면 종종 FileNotFound 가 나거나 엉뚱한 곳을 가리킨다(되는 경우도 있어 일관성이 없음).

**Why:** 2026-11-04 바탕화면의 작업 폴더에 파일을 복사·생성하다 이 문제를 겪었다. 사용자 폴더 옆에 보이는 깨진 이름의 폴더들은 그래픽 드라이버 로그가 만든 기존 것이지 우리가 만든 게 아니다.

**How to apply:** 한글이 든 경로는 PowerShell 툴(`[Environment]::GetFolderPath('Desktop')`)이나 파이썬에서 `ctypes.windll.shell32.SHGetFolderPathW` 로 얻어서 쓰고, Write/Edit 툴은 그대로 동작한다. 저장소와 .venv 는 Bash 로 문제없음. 파이썬 출력의 한글 mojibake 는 콘솔 인코딩 문제일 뿐 파일 내용은 정상.
