param(
    [string]$HostName = "149.165.152.254",
    [string]$RemoteDir = "/media/volume/AdditionalHeadroom/Txn_Jatin_studio_20260726",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path (Split-Path -Parent $Root) ".venv\Scripts\python.exe"
$Uploader = Join-Path (Split-Path -Parent $Root) "Txn_Jatin_20epochs\remote_upload.py"
$Executor = Join-Path (Split-Path -Parent $Root) "Txn_Jatin_20epochs\remote_exec.py"

if (-not $env:REMOTE_PASS) {
    throw "REMOTE_PASS must be set for the remote SSH account."
}

$Archive = Join-Path $env:TEMP "txn_jatin_studio_deploy.tar.gz"
tar -czf $Archive -C $Root app config results
& $Python $Uploader --host $HostName --remote-dir $RemoteDir $Archive
& $Python $Executor --host $HostName --timeout 120 -- "set -e; cd $RemoteDir; tar -xzf txn_jatin_studio_deploy.tar.gz; if [ -s studio.pid ]; then old=`$(cat studio.pid); if kill -0 `$old 2>/dev/null && ps -p `$old -o args= | grep -q 'app/backend/server.py'; then kill `$old; for i in 1 2 3 4 5; do kill -0 `$old 2>/dev/null || break; sleep 1; done; fi; fi; nohup /media/volume/TrainingData/home_data/benchmarking_run/.venv/bin/python app/backend/server.py --port $Port > studio.log 2>&1 < /dev/null & pid=`$!; echo `$pid > studio.pid; sleep 3; kill -0 `$pid; curl -fsS http://127.0.0.1:$Port/api/experiments >/dev/null; echo STUDIO_HEALTH_OK"
Write-Host "Remote Studio: http://${HostName}:$Port/"
