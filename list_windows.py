#!/usr/bin/env python3
"""List all open windows to help debug window detection."""
import subprocess
import sys

try:
    result = subprocess.run(['xdotool', 'search', '--name', '.'], capture_output=True, text=True)
    if result.returncode != 0:
        print("xdotool not found. Install with: sudo apt install xdotool")
        sys.exit(1)
    
    winids = result.stdout.strip().splitlines()
    print(f"Found {len(winids)} windows:\n")
    
    for winid in winids:
        name_result = subprocess.run(['xdotool', 'getwindowname', winid], capture_output=True, text=True)
        name = name_result.stdout.strip()
        if name:
            print(f"  ID: {winid:12} | Name: {name}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
