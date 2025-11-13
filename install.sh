#!/usr/bin/env bash
#
# Setup script for the found-tools development environment.
# This script installs dependencies for Linux (Debian/Fedora-based) and macOS.

# Exit on any command failure
set -e

# Executes a command with a descriptive banner
execute_cmd() {
    local cmd="$@"
    local middle_line="Command: $cmd"
    local middle_line_len=${#middle_line}
    local extra_chars=20
    local total_length=$((middle_line_len + extra_chars))
    local hash_line
    hash_line=$(printf '%*s' "$total_length" | tr ' ' '=')

    printf "\n%s\n" "$hash_line"
    printf "          %s\n" "$middle_line"
    printf "%s\n" "$hash_line\n"

    # Execute the command and exit if it fails
    if ! eval "$cmd"; then
        echo "Command failed: $cmd"
        exit 1
    fi
}

# --- Main Setup ---

# Detect if running as root (UID 0)
if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Detect the operating system
OS="$(uname -s)"
echo "Detected Operating System: $OS"

# Perform setup based on the detected OS
case "$OS" in
    Linux*)
        # --- Linux Setup ---
        if command -v apt-get &> /dev/null; then
            PM="$SUDO apt-get"
            execute_cmd "$PM -y update"
            INSTALL_CMD="$PM install -y"
        else
            echo "No supported package manager (apt-get) found."
            exit 1
        fi

        # Install base dependencies and pipx
        execute_cmd "$INSTALL_CMD git curl python3 python3-pip python3-venv pipx"

        # Install core tools with pipx
        execute_cmd "pipx install uv"
        execute_cmd "pipx install rust-just"
        execute_cmd "pipx ensurepath" # Ensure tools are in PATH
        ;;

    Darwin*)
        # --- macOS Setup ---
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Installing Homebrew..."
            execute_cmd '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        fi
        
        # Install core tools with Homebrew
        execute_cmd "brew install uv just"
        ;;

    *)
        echo "Unsupported Operating System: $OS"
        exit 1
        ;;
esac

# --- Project Setup ---

# Install Python tools required for the project using uv
echo "Installing project-specific tools (ruff)..."
execute_cmd "uv tool install ruff"

# Sync Python dependencies from pyproject.toml
echo "Syncing project dependencies..."
execute_cmd "uv sync"

# --- Finalization ---
printf "\n============ Setup Complete ========================\n"
printf "Please restart your terminal to apply PATH changes.\n"
printf "After restarting, you can run 'just check-all' to verify the setup.\n"
