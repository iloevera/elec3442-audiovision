#!/bin/bash
# setup_autostart.sh - Script to setup main.py to run on startup via systemd

PROJECT_DIR=$(pwd)
SERVICE_NAME="audiovision.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
PYTHON_VENV="$PROJECT_DIR/.venv/bin/python3"
USER_NAME=$(whoami)

echo "Setting up $SERVICE_NAME to run $PROJECT_DIR/main.py on startup..."

# Check if .venv exists
if [ ! -f "$PYTHON_VENV" ]; then
    echo "Error: Virtual environment not found at $PYTHON_VENV"
    echo "Please ensure you have created the .venv folder."
    exit 1
fi

# Create service file
cat <<EOF | sudo tee $SERVICE_PATH
[Unit]
Description=Audiovision Main Service
After=network.target

[Service]
ExecStart=$PYTHON_VENV $PROJECT_DIR/main.py
WorkingDirectory=$PROJECT_DIR
StandardOutput=inherit
StandardError=inherit
Restart=always
User=$USER_NAME

[Install]
WantedBy=multi-user.target
EOF

# Reload, enable and start
echo "Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "Setup complete. Service status:"
sudo systemctl status $SERVICE_NAME --no-pager
