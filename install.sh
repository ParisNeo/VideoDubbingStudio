#!/bin/bash
set -e

# VoiceDub Pro - Linux/macOS Installation Script
# Downloads and sets up FFmpeg locally in the project folder

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_step() {
    echo ""
    echo -e "${CYAN}[$1/5] $2${NC}"
    echo "------------------------------------------------------------"
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get OS type
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists lsb_release; then
            OS=$(lsb_release -si)
        elif [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$ID
        else
            OS="linux"
        fi
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS_TYPE=$(detect_os)

# -------------------------------------------------------------------------------------------------------------------------------
# STEP 1: Check Python
# -------------------------------------------------------------------------------------------------------------------------------
log_step "1" "Checking Python..."

if ! command_exists python3; then
    log_error "python3 could not be found."
    log_info "Please install Python 3.9+ from https://python.org"
    exit 1
fi

PY_VERSION=$(python3 --version 2>&1)
log_info "Found: $PY_VERSION"

# Extract version numbers and check >= 3.9
PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    log_error "Python 3.9+ required, found $PY_MAJOR.$PY_MINOR"
    exit 1
fi

# -------------------------------------------------------------------------------------------------------------------------------
# STEP 2: Create Virtual Environment
# -------------------------------------------------------------------------------------------------------------------------------
log_step "2" "Creating Virtual Environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "venv created."
else
    log_warn "venv already exists."
fi

# -------------------------------------------------------------------------------------------------------------------------------
# STEP 3: Install Python Dependencies
# -------------------------------------------------------------------------------------------------------------------------------
log_step "3" "Installing Python Libraries..."

source venv/bin/activate

pip install --upgrade pip

if [ ! -f "requirements.txt" ]; then
    log_error "requirements.txt not found in current directory"
    exit 1
fi

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    log_error "pip install failed"
    exit 1
fi

log_success "Python dependencies installed."

# -------------------------------------------------------------------------------------------------------------------------------
# STEP 4: Download FFmpeg
# -------------------------------------------------------------------------------------------------------------------------------
log_step "4" "Downloading Portable FFmpeg..."

FFMPEG_DIR="$SCRIPT_DIR/ffmpeg_local"
mkdir -p "$FFMPEG_DIR"

# Check if we already have working ffmpeg
check_existing_ffmpeg() {
    if [ -f "$SCRIPT_DIR/ffmpeg" ]; then
        if "$SCRIPT_DIR/ffmpeg" -version >/dev/null 2>&1; then
            local version=$("$SCRIPT_DIR/ffmpeg" -version 2>&1 | head -n 1)
            log_success "FFmpeg already exists: $version"
            return 0
        else
            log_warn "Existing ffmpeg is broken, will re-download"
            rm -f "$SCRIPT_DIR/ffmpeg" "$SCRIPT_DIR/ffprobe"
            return 1
        fi
    fi
    return 1
}

if ! check_existing_ffmpeg; then
    log_info "OS detected: $OS_TYPE"
    
    case $OS_TYPE in
        linux)
            # Download static build from johnvansickle.com (most reliable)
            FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
            FFMPEG_ARCHIVE="ffmpeg-linux-amd64.tar.xz"
            
            log_info "Downloading Linux static build from $FFMPEG_URL..."
            
            if command_exists curl; then
                curl -L -o "$FFMPEG_ARCHIVE" "$FFMPEG_URL" --progress-bar
            elif command_exists wget; then
                wget --progress=bar:force -O "$FFMPEG_ARCHIVE" "$FFMPEG_URL"
            else
                log_error "Neither curl nor wget found. Cannot download FFmpeg."
                exit 1
            fi
            
            log_info "Extracting..."
            tar -xf "$FFMPEG_ARCHIVE"
            
            # Find the extracted directory (it has version in name)
            EXTRACTED_DIR=$(find . -maxdepth 1 -type d -name "ffmpeg-*-static" | head -n 1)
            
            if [ -z "$EXTRACTED_DIR" ]; then
                log_error "Could not find extracted FFmpeg directory"
                rm -f "$FFMPEG_ARCHIVE"
                exit 1
            fi
            
            # Move binaries to project root
            mv "$EXTRACTED_DIR/ffmpeg" "$SCRIPT_DIR/ffmpeg"
            mv "$EXTRACTED_DIR/ffprobe" "$SCRIPT_DIR/ffprobe"
            
            # Cleanup
            rm -rf "$EXTRACTED_DIR"
            rm -f "$FFMPEG_ARCHIVE"
            ;;
            
        macos)
            # For macOS, we need to handle both Intel and Apple Silicon
            ARCH=$(uname -m)
            
            if [ "$ARCH" = "arm64" ]; then
                # Apple Silicon - use homebrew or native build
                log_info "Apple Silicon (M1/M2/M3) detected"
                
                # Try to use homebrew if available
                if command_exists brew; then
                    log_info "Homebrew detected. You can install FFmpeg via: brew install ffmpeg"
                    log_warn "Downloading pre-built binary for Apple Silicon..."
                    
                    # Use evermeet.cx builds (native Apple Silicon)
                    FFMPEG_URL="https://evermeet.cx/ffmpeg/ffmpeg-6.1.zip"
                    FFPROBE_URL="https://evermeet.cx/ffmpeg/ffprobe-6.1.zip"
                else
                    log_warn "Homebrew not found. Using Intel build with Rosetta..."
                    
                    # Fallback to Intel build
                    FFMPEG_URL="https://evermeet.cx/ffmpeg/ffmpeg-6.1.zip"
                    FFPROBE_URL="https://evermeet.cx/ffmpeg/ffprobe-6.1.zip"
                fi
            else
                # Intel Mac
                log_info "Intel Mac detected"
                FFMPEG_URL="https://evermeet.cx/ffmpeg/ffmpeg-6.1.zip"
                FFPROBE_URL="https://evermeet.cx/ffmpeg/ffprobe-6.1.zip"
            fi
            
            log_info "Downloading macOS build..."
            
            # Download ffmpeg
            if command_exists curl; then
                curl -L -o "ffmpeg.zip" "$FFMPEG_URL"
                curl -L -o "ffprobe.zip" "$FFPROBE_URL"
            else
                wget -O "ffmpeg.zip" "$FFMPEG_URL"
                wget -O "ffprobe.zip" "$FFPROBE_URL"
            fi
            
            log_info "Extracting..."
            unzip -o "ffmpeg.zip"
            unzip -o "ffprobe.zip"
            
            # Make executable
            chmod +x ffmpeg ffprobe
            
            # Cleanup
            rm -f "ffmpeg.zip" "ffprobe.zip"
            ;;
            
        *)
            log_error "Unsupported OS: $OSTYPE"
            log_info "Please install FFmpeg manually from https://ffmpeg.org/download.html"
            exit 1
            ;;
    esac
    
    # Verify installation
    if [ -f "$SCRIPT_DIR/ffmpeg" ]; then
        if "$SCRIPT_DIR/ffmpeg" -version >/dev/null 2>&1; then
            VERSION=$("$SCRIPT_DIR/ffmpeg" -version 2>&1 | head -n 1)
            log_success "FFmpeg installed and verified: $VERSION"
        else
            log_error "FFmpeg download succeeded but binary doesn't work"
            log_info "You may need to install FFmpeg manually"
            rm -f "$SCRIPT_DIR/ffmpeg" "$SCRIPT_DIR/ffprobe"
        fi
    else
        log_error "FFmpeg installation failed - binary not found"
        exit 1
    fi
fi

# Make absolutely sure binaries are executable
chmod +x "$SCRIPT_DIR/ffmpeg" "$SCRIPT_DIR/ffprobe" 2>/dev/null || true

# -------------------------------------------------------------------------------------------------------------------------------
# STEP 5: Finalize Installation
# -------------------------------------------------------------------------------------------------------------------------------
log_step "5" "Finalizing Installation..."

# Create necessary directories
for dir in uploads outputs temp_chunks projects_db model_cache static templates; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        log_info "Created: $dir/"
    fi
done

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}     VoiceDub Pro Installation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "To run the application:"
echo ""
echo -e "${CYAN}  1. Activate virtual environment:${NC}"
echo "     source venv/bin/activate"
echo ""
echo -e "${CYAN}  2. Start the server:${NC}"
echo "     python server.py"
echo ""
echo -e "${CYAN}  3. Open in browser:${NC}"
echo "     http://localhost:8000"
echo ""
echo -e "${CYAN}FFmpeg location:${NC} $SCRIPT_DIR/ffmpeg"
echo ""
