@echo off
rem Open images/ in labelme for review (autosave on; D = next, A = prev, Ctrl+N = new polygon).
rem Uses the repo .venv if present (dataset_own\..\.venv), otherwise `labelme` on PATH.
set "VENV=%~dp0..\.venv\Scripts"
if exist "%VENV%\labelme.exe" (
  "%VENV%\labelme.exe" "%~dp0images" --labels "%~dp0labels.txt" --nodata --autosave
) else (
  labelme "%~dp0images" --labels "%~dp0labels.txt" --nodata --autosave
)
