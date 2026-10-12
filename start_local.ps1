Write-Host "=========================================="
Write-Host "    CheckReality AI - Unified Local Start"
Write-Host "=========================================="
Write-Host "Compiling Frontend..."
Set-Location .\frontend
npm run build
Set-Location ..

Write-Host "Starting Backend & Worker in Background..."
Set-Location .\backend

if (-Not (Test-Path ".\venv")) {
    Write-Host "Virtual environment not found. Creating venv and installing dependencies..."
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
}

# Launch the celery worker in a separate window but linked to this process
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location .\backend; .\venv\Scripts\Activate.ps1; celery -A worker.celery_app worker --loglevel=info --pool=solo" -WindowStyle Minimized

# Launch the API which now serves the built frontend on port 8001
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --port 8001
