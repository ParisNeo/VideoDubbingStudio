# VoiceDub Pro - Windows Installation Script
# Downloads and sets up FFmpeg locally in the project folder

param(
    [switch]$SkipVenv,
    [switch]$SkipFFmpeg
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message, [int]$Step, [int]$Total)
    Write-Host ""
    Write-Host "[$Step/$Total] $Message" -ForegroundColor Cyan
    Write-Host ("-" * 60) -ForegroundColor DarkGray
}

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# =============================================================================
# STEP 1: Check Python
# =============================================================================
Write-Step "Checking Python..." 1 5

if (-not (Test-Command "python")) {
    if (Test-Command "python3") {
        $pythonCmd = "python3"
    } else {
        Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
        Write-Host "Please install Python 3.9+ from https://python.org"
        exit 1
    }
} else {
    $pythonCmd = "python"
}

# Check Python version
$pyVersion = & $pythonCmd --version 2>&1
Write-Host "Found: $pyVersion"

if ($pyVersion -match "Python (\d+)\.(\d+)") {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
        Write-Host "ERROR: Python 3.9+ required, found $major.$minor" -ForegroundColor Red
        exit 1
    }
}

# =============================================================================
# STEP 2: Create Virtual Environment
# =============================================================================
Write-Step "Creating Virtual Environment..." 2 5

if (-not $SkipVenv) {
    if (-not (Test-Path "venv")) {
        & $pythonCmd -m venv venv
        Write-Host "Created venv/" -ForegroundColor Green
    } else {
        Write-Host "venv/ already exists, skipping" -ForegroundColor Yellow
    }
    
    # Activate venv for subsequent commands
    $venvPython = ".\venv\Scripts\python.exe"
    $venvPip = ".\venv\Scripts\pip.exe"
    
    if (-not (Test-Path $venvPython)) {
        Write-Host "ERROR: Virtual environment creation failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Skipping venv creation (--SkipVenv specified)" -ForegroundColor Yellow
    $venvPython = $pythonCmd
    $venvPip = if (Test-Command "pip") { "pip" } else { "$pythonCmd -m pip" }
}

# =============================================================================
# STEP 3: Install Python Dependencies
# =============================================================================
Write-Step "Installing Python Libraries..." 3 5

& $venvPip install --upgrade pip

$reqFile = "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Write-Host "ERROR: $reqFile not found in current directory" -ForegroundColor Red
    exit 1
}

& $venvPip install -r $reqFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed" -ForegroundColor Red
    exit 1
}

Write-Host "Python dependencies installed" -ForegroundColor Green

# =============================================================================
# STEP 4: Download FFmpeg for Windows
# =============================================================================
Write-Step "Downloading Portable FFmpeg..." 4 5

if ($SkipFFmpeg) {
    Write-Host "Skipping FFmpeg download (--SkipFFmpeg specified)" -ForegroundColor Yellow
} else {
    # Check if FFmpeg already exists and works
    $ffmpegExists = $false
    if (Test-Path "ffmpeg.exe") {
        try {
            $ffVersion = & ".\ffmpeg.exe" -version 2>&1 | Select-Object -First 1
            if ($ffVersion -match "ffmpeg version") {
                Write-Host "FFmpeg already exists: $ffVersion" -ForegroundColor Green
                $ffmpegExists = $true
            }
        } catch {
            Write-Host "Existing ffmpeg.exe is broken, will re-download" -ForegroundColor Yellow
            Remove-Item "ffmpeg.exe" -Force -ErrorAction SilentlyContinue
            Remove-Item "ffprobe.exe" -Force -ErrorAction SilentlyContinue
        }
    }

    if (-not $ffmpegExists) {
        # Determine architecture
        $arch = "x64"  # Default to 64-bit
        if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
            $arch = "arm64"
        }

        # FFmpeg builds from BtbN (reliable, up-to-date)
        # Using latest release build
        $baseUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
        
        # Windows builds
        $ffmpegZip = "ffmpeg-master-latest-win64-gpl.zip"
        $downloadUrl = "$baseUrl/$ffmpegZip"

        Write-Host "Downloading FFmpeg for Windows ($arch)..."
        Write-Host "URL: $downloadUrl"

        $tempDir = [System.IO.Path]::GetTempPath()
        $zipPath = Join-Path $tempDir $ffmpegZip
        $extractDir = Join-Path $tempDir "ffmpeg_extract"

        try {
            # Clean up any old downloads
            Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
            Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue

            # Download using curl.exe (avoiding PowerShell alias issues)
            Write-Host "Downloading... (this may take a minute)"
            $curlCmd = "curl.exe -L -o `"$zipPath`" `"$downloadUrl`""
            Invoke-Expression $curlCmd
            
            if (-not (Test-Path $zipPath)) {
                throw "Download failed - file not created"
            }

            $fileSize = (Get-Item $zipPath).Length / 1MB
            Write-Host "Downloaded: $([math]::Round($fileSize, 2)) MB"

            # Extract using PowerShell's Expand-Archive
            Write-Host "Extracting..."
            Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

            # Find the ffmpeg.exe in extracted folder (handles varying folder names)
            $ffmpegSource = Get-ChildItem -Path $extractDir -Recurse -Filter "ffmpeg.exe" | 
                Select-Object -First 1
            $ffprobeSource = Get-ChildItem -Path $extractDir -Recurse -Filter "ffprobe.exe" | 
                Select-Object -First 1

            if (-not $ffmpegSource) {
                throw "ffmpeg.exe not found in extracted archive"
            }

            # Copy to project root
            Copy-Item $ffmpegSource.FullName -Destination "ffmpeg.exe" -Force
            Write-Host "Installed: ffmpeg.exe" -ForegroundColor Green

            if ($ffprobeSource) {
                Copy-Item $ffprobeSource.FullName -Destination "ffprobe.exe" -Force
                Write-Host "Installed: ffprobe.exe" -ForegroundColor Green
            }

            # Verify installation
            $verifyVersion = & ".\ffmpeg.exe" -version 2>&1 | Select-Object -First 1
            Write-Host "Verified: $verifyVersion" -ForegroundColor Green

        } catch {
            Write-Host ""
            Write-Host "ERROR: FFmpeg download/install failed: $_" -ForegroundColor Red
            Write-Host ""
            Write-Host "Falling back to manual installation instructions:" -ForegroundColor Yellow
            Write-Host "1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases"
            Write-Host "2. Extract ffmpeg.exe and ffprobe.exe to this folder"
            Write-Host ""
            
            # Don't exit - the app might still work if user has system FFmpeg
            Write-Host "Continuing without local FFmpeg..." -ForegroundColor Yellow
        } finally {
            # Cleanup
            Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
            Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# =============================================================================
# STEP 5: Finalize Installation
# =============================================================================
Write-Step "Finalizing Installation..." 5 5

# Create necessary directories
$dirs = @("uploads", "outputs", "temp_chunks", "projects_db", "model_cache", "static", "templates")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created: $dir/" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  VoiceDub Pro Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:"
Write-Host ""
Write-Host "  1. Activate virtual environment:" -ForegroundColor Cyan
Write-Host "     .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  2. Start the server:" -ForegroundColor Cyan
Write-Host "     python server.py"
Write-Host ""
Write-Host "  3. Open in browser:" -ForegroundColor Cyan
Write-Host "     http://localhost:8000"
Write-Host ""
Write-Host "FFmpeg location: $(Get-Location)\ffmpeg.exe" -ForegroundColor DarkGray
Write-Host ""
