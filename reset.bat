@echo off
echo Resetting pipeline state...

rmdir /s /q output
mkdir output

del state\visited_pages.csv >nul 2>&1
del state\completed_topics.csv >nul 2>&1
del state\completed_requests.csv >nul 2>&1

echo url > state\visited_pages.csv
echo key > state\completed_topics.csv
echo key > state\completed_requests.csv

echo Reset complete. Ready for fresh run.
pause