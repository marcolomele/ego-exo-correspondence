#!/usr/bin/env python3
"""
Upload directories to Google Drive using rclone, with subfolder-level tqdm progress bar.

Usage:
    python upload_to_drive.py --source {source directory} --dest {destination directory}
"""

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import sys


def upload_to_drive(source_dir: str, dest_path: str, remote_name: str = "remote"):
    """
    Upload directory to Google Drive using rclone, with tqdm subfolder-level progress.
    
    Args:
        source_dir: Local directory to upload
        dest_path: Destination path on Google Drive
        remote_name: rclone remote name (default: remote)
    """
    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    
    remote_dest = f"{remote_name}:{dest_path}"

    # Get all first-level subdirectories and files
    subfolders = [p for p in source.iterdir() if p.is_dir()]
    files = [p for p in source.iterdir() if p.is_file()]

    if not subfolders and not files:
        print("Nothing to upload: directory is empty.")
        return

    print(f"Uploading {source.name} → {remote_dest}\n")

    with tqdm(total=len(subfolders) + (1 if files else 0), desc="Upload Progress", unit="item", ncols=100) as pbar:
        # Upload files directly in the source directory (not inside subfolders)
        if files:
            cmd = [
                "rclone", "sync",
                str(source),
                remote_dest,
                "--transfers", "128",  # Maximum parallelism for high bandwidth
                "--checkers", "128",  # More parallel checks
                "--drive-chunk-size", "32M",  # Very large chunks for fewer API calls
                "--drive-upload-cutoff", "32M",  # Resumable uploads for large files
                "--buffer-size", "128M",  # Large buffer to saturate bandwidth
                "--fast-list",  # Use Drive API optimization
                "--drive-use-trash=false",  # Skip trash (faster)
                "--low-level-retries", "10",  # Retry on transient errors
                "--stats", "0",  # Disable stats output
                "--exclude", "*/"  # Only copy files in this directory, not subdirs
            ]
            try:
                # Suppress rclone output, only show errors
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                pbar.update(1)
                print(f" Creted {source.name}")
            except subprocess.CalledProcessError as e:
                # Show error if upload fails
                error_msg = e.stderr.decode() if e.stderr else str(e)
                print(f"\n✗ Upload failed for files in {source.name}")
                print(f"Error: {error_msg}")
                raise

        # Upload each first-level subfolder independently (for tqdm progress)
        for sub in subfolders:
            rel_dest = str(Path(dest_path) / sub.name)
            cmd = [
                "rclone", "sync",
                str(sub),
                f"{remote_name}:{rel_dest}",
                "--transfers", "128",  # Maximum parallelism for high bandwidth
                "--checkers", "128",  # More parallel checks
                "--drive-chunk-size", "32M",  # Very large chunks for fewer API calls
                "--drive-upload-cutoff", "32M",  # Resumable uploads for large files
                "--buffer-size", "128M",  # Large buffer to saturate bandwidth
                "--fast-list",  # Use Drive API optimization
                "--drive-use-trash=false",  # Skip trash (faster)
                "--low-level-retries", "10",  # Retry on transient errors
                "--stats", "0"  # Disable stats output
            ]
            try:
                # Suppress rclone output, only show errors
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                pbar.update(1)
                print(f" Uploaded: {sub.name}")
            except subprocess.CalledProcessError as e:
                # Show error if upload fails
                error_msg = e.stderr.decode() if e.stderr else str(e)
                print(f"\n✗ Upload failed for {sub.name}")
                print(f"Error: {error_msg}")
                raise

    print("\n✓ All uploads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload directory to Google Drive via rclone")
    parser.add_argument("--source", required=True, help="Source directory to upload")
    parser.add_argument("--dest", required=True, help="Destination path on Google Drive")
    parser.add_argument("--remote", default="remote", help="rclone remote name (default: remote)")

    args = parser.parse_args()
    try:
        upload_to_drive(args.source, args.dest, args.remote)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
