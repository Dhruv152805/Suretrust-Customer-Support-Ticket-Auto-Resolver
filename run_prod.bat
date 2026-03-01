@echo off
echo Starting Customer Support Ticket Auto-Resolver API in Production Mode...
set PYTHONPATH=.
uvicorn src.app:app --host 127.0.0.1 --port 8000 --workers 4 --no-access-log
pause
