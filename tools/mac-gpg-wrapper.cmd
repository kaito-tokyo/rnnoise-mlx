@echo off
setlocal
set "SCRIPT=/mnt/c/Users/umireon/.local/bin/mac-gpg-wrapper.sh"
wsl.exe bash -lc "tr -d '\r' < '%SCRIPT%' > /tmp/mac-gpg-wrapper.sh && exec bash /tmp/mac-gpg-wrapper.sh mac-gpg-wrapper %*"
exit /b %ERRORLEVEL%
