@echo off
cd /d C:\ユーザー_Harada\Claude\app1
echo シグナルを計算中...
python sector_momentum.py
echo GitHubにアップロード中...
git add signal_dashboard.html
git commit -m "update signal"
git push
echo.
echo 完了しました。
pause
