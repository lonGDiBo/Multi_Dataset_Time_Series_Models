#!/bin/bash
apt-get update
apt-get install -y python3-distutils
pip install --no-cache-dir -r requirements.txt

chmod +x install.sh