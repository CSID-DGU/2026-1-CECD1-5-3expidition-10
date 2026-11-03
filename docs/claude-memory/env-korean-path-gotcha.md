---
name: env-korean-path-gotcha
description: 이 PC 는 사용자 폴더가 한글(C:\Users\우리집컴)이라 Git Bash 에서 Desktop 경로를 넘기면 깨진다 — PowerShell 이나 ctypes 로 경로를 얻을 것
metadata:
  type: feedback
---

사용자 프로필이 `C:\Users\우리집컴` 이라 Bash 툴(Git Bash)로 `C:/Users/우리집컴/...` 또는 `/c/Users/우리집컴/Desktop/...` 를 cp/python 인자로 넘기면 종종 FileNotFound 가 나거나 엉뚱한 곳을 가리킨다(스크래치패드 `/c/Users/우리집컴/AppData/...` 는 되는 경우도 있어 일관성이 없음).

**Why:** 2026-11-04 Desktop\dataset_own 에 파일을 복사·생성하다 같은 문제를 겪었다. C:\Users 아래의 깨진 이름 폴더(`�츮����`, `﾿￬ﾸﾮ￁�ￄￄ`)는 AMD/NVIDIA 로그가 만든 기존 것이지 우리가 만든 게 아니다.

**How to apply:** Desktop 등 한글 경로는 PowerShell 툴(`[Environment]::GetFolderPath('Desktop')`)이나 파이썬에서 `ctypes.windll.shell32.SHGetFolderPathW` 로 얻어서 쓰고, Write/Edit 툴은 그대로 동작한다. 저장소(D:\Program Files\...)와 .venv 는 Bash 로 문제없음. 파이썬 출력의 한글 mojibake 는 콘솔 인코딩 문제일 뿐 파일 내용은 정상.
