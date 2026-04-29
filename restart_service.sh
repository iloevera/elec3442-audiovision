#!/bin/bash
# Restart the audiovision systemd service
echo "Restarting audiovision service..."
sudo systemctl restart audiovision.service
echo "Service status:"
sudo systemctl status audiovision.service --no-pager
