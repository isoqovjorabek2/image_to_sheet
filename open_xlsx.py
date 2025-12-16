#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script to open XLSX files.
Can open with default application or display contents in terminal.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
from pathlib import Path


def open_with_default_app(file_path: str):
    """Open file with system's default application."""
    file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    try:
        if sys.platform == 'win32':
            # Windows
            os.startfile(file_path)
        elif sys.platform == 'darwin':
            # macOS
            subprocess.run(['open', file_path])
        else:
            # Linux
            subprocess.run(['xdg-open', file_path])
        print(f"Opening {file_path} with default application...")
        return True
    except Exception as e:
        print(f"Error opening file: {e}")
        return False


def display_in_terminal(file_path: str, max_rows: int = 50):
    """Display XLSX file contents in terminal."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        print(f"\n{'='*80}")
        print(f"Contents of: {file_path}")
        print(f"{'='*80}\n")
        
        # Display shape
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
        
        # Display first few rows
        print(df.head(max_rows).to_string())
        
        if len(df) > max_rows:
            print(f"\n... (showing first {max_rows} of {len(df)} rows)")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Open or display XLSX files'
    )
    parser.add_argument(
        'file',
        type=str,
        help='Path to the XLSX file'
    )
    parser.add_argument(
        '-d', '--display',
        action='store_true',
        help='Display contents in terminal instead of opening with default app'
    )
    parser.add_argument(
        '-r', '--rows',
        type=int,
        default=50,
        help='Maximum number of rows to display (default: 50)'
    )
    
    args = parser.parse_args()
    
    if args.display:
        display_in_terminal(args.file, args.rows)
    else:
        open_with_default_app(args.file)


if __name__ == '__main__':
    main()

