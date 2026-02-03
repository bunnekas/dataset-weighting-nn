#!/usr/bin/env python3
"""
Single-script OpenImages downloader with status tracking.
Downloads images until target count is reached, logging all attempts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict

import boto3
import botocore
from tqdm import tqdm


class DownloadStatus(Enum):
    """Status of download attempts"""
    SUCCESS = "success"
    EXISTS = "exists"
    NOT_FOUND = "not found"
    FORBIDDEN = "forbidden"
    OTHER_ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class DownloadResult:
    """Result of a single download attempt"""
    split: str
    image_id: str
    status: DownloadStatus
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: Optional[str] = None
    filename: Optional[str] = None
    retry_count: int = 0


class OpenImagesDownloader:
    """Download OpenImages with comprehensive logging"""
    
    BUCKET_NAME = "open-images-dataset"
    
    def __init__(
        self,
        csv_path: Path,
        target_count: int,
        out: Path,
        split: str = "train",
        seed: int = 0,
        num_workers: int = 32,
        max_retries: int = 0
    ):
        self.csv_path = csv_path
        self.target_count = target_count
        self.out = out
        self.split = split
        self.seed = seed
        self.num_workers = num_workers
        self.max_retries = max_retries
        
        # Set up paths relative to output root
        self.images_dir = self.out / "images"
        self.log_file = self.out / "download_log.jsonl"
        self.success_manifest = self.out / "success_manifest.txt"
        self.summary_file = self.out / "download_summary.json"
        
        # Initialize tracking
        self.downloaded_count = 0
        self.total_attempted = 0
        self.results: List[DownloadResult] = []
        self.successful_ids: List[str] = []
        
        # S3 client for public bucket
        self.s3 = boto3.resource(
            "s3",
            config=botocore.config.Config(signature_version=botocore.UNSIGNED),
        )
        self.bucket = self.s3.Bucket(self.BUCKET_NAME)
        
    def load_image_ids(self) -> List[tuple[str, str]]:
        """Load image IDs from CSV file with split filtering"""
        print(f"Loading image IDs from {self.csv_path}...")
        
        # Try to detect the right columns
        with self.csv_path.open('r') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV has no header row")
            
            # Look for common column names
            col_names = [name.lower() for name in reader.fieldnames]
            
            # Find split column
            split_col = None
            for name in ['split', 'subset', 'set']:
                if name in col_names:
                    split_col = reader.fieldnames[col_names.index(name)]
                    break
            
            # Find image ID column
            id_col = None
            for name in ['imageid', 'image_id', 'id', 'image']:
                if name in col_names:
                    id_col = reader.fieldnames[col_names.index(name)]
                    break
            
            if not split_col or not id_col:
                available = ", ".join(reader.fieldnames)
                raise ValueError(
                    f"Could not detect required columns. Available: {available}\n"
                    f"Need 'Split' (or similar) and 'ImageID' (or similar)."
                )
            
            # Collect IDs for the specified split
            ids = []
            for row in reader:
                if row[split_col].strip().lower() == self.split.lower():
                    img_id = row[id_col].strip()
                    if img_id:
                        ids.append((self.split, img_id))
        
        print(f"Found {len(ids)} images in split '{self.split}'")
        return ids
    
    def attempt_download(
        self, 
        split: str, 
        image_id: str,
        retry_count: int = 0
    ) -> DownloadResult:
        """Attempt to download a single image with retry logic"""
        timestamp = datetime.now().isoformat()
        
        # Create output filename: just image_id.jpg (no split prefix)
        output_path = self.images_dir / f"{image_id}.jpg"
        
        # Check if already exists
        if output_path.exists():
            return DownloadResult(
                split=split,
                image_id=image_id,
                status=DownloadStatus.EXISTS,
                filename=str(output_path.name),
                timestamp=timestamp,
                retry_count=retry_count
            )
        
        # Try different key patterns
        key_patterns = [
            f"{split}/{image_id}.jpg",
            f"{split}/{image_id}",
        ]
        
        for key in key_patterns:
            try:
                # Download with timeout
                tmp_path = output_path.with_suffix(".tmp")
                self.bucket.download_file(key, str(tmp_path))
                
                # Rename temp file to final name
                tmp_path.rename(output_path)
                
                return DownloadResult(
                    split=split,
                    image_id=image_id,
                    status=DownloadStatus.SUCCESS,
                    filename=str(output_path.name),
                    timestamp=timestamp,
                    retry_count=retry_count
                )
                
            except botocore.exceptions.ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_msg = str(e)
                
                # If it's a 404, try next pattern
                if error_code in ['404', 'NoSuchKey']:
                    continue
                else:
                    # Other errors (403, 500, etc.)
                    return DownloadResult(
                        split=split,
                        image_id=image_id,
                        status=DownloadStatus.FORBIDDEN if error_code == '403' else DownloadStatus.OTHER_ERROR,
                        error_code=error_code,
                        error_message=error_msg,
                        timestamp=timestamp,
                        retry_count=retry_count
                    )
            except Exception as e:
                # Network or other errors
                return DownloadResult(
                    split=split,
                    image_id=image_id,
                    status=DownloadStatus.OTHER_ERROR,
                    error_code="NetworkError",
                    error_message=str(e),
                    timestamp=timestamp,
                    retry_count=retry_count
                )
        
        # All patterns failed with 404
        return DownloadResult(
            split=split,
            image_id=image_id,
            status=DownloadStatus.NOT_FOUND,
            error_code="404",
            error_message="All key patterns failed",
            timestamp=timestamp,
            retry_count=retry_count
        )
    
    def log_result(self, result: DownloadResult):
        """Log individual result to JSONL file"""
        log_entry = asdict(result)
        log_entry['status'] = result.status.value
        
        with self.log_file.open('a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Track successes
        if result.status in [DownloadStatus.SUCCESS, DownloadStatus.EXISTS]:
            self.successful_ids.append(f"{result.split}/{result.image_id}")
            self.downloaded_count += 1
    
    def write_summary(self):
        """Write summary of download results"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "csv_file": str(self.csv_path),
                "target_count": self.target_count,
                "out": str(self.out),
                "split": self.split,
                "seed": self.seed,
                "num_workers": self.num_workers,
                "max_retries": self.max_retries,
            },
            "results": {
                "total_attempted": self.total_attempted,
                "successful_downloads": self.downloaded_count,
                "target_reached": self.downloaded_count >= self.target_count,
                "by_status": {},
            },
            "outputs": {
                "images_directory": str(self.images_dir),
                "log_file": str(self.log_file),
                "success_manifest": str(self.success_manifest),
                "summary_file": str(self.summary_file),
            }
        }
        
        # Count by status
        for result in self.results:
            status = result.status.value
            summary["results"]["by_status"][status] = \
                summary["results"]["by_status"].get(status, 0) + 1
        
        # Write summary JSON
        with self.summary_file.open('w') as f:
            json.dump(summary, f, indent=2)
        
        # Write success manifest
        with self.success_manifest.open('w') as f:
            for img_id in self.successful_ids[:self.target_count]:
                f.write(f"{img_id}\n")
    
    def print_progress(self):
        """Print progress summary"""
        status_counts: Dict[str, int] = {}
        for result in self.results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\n" + "="*60)
        print("DOWNLOAD PROGRESS")
        print("="*60)
        print(f"Attempted: {self.total_attempted}")
        print(f"Successful: {self.downloaded_count}/{self.target_count}")
        print(f"Retries configured: {self.max_retries}")
        print("-"*60)
        for status, count in sorted(status_counts.items()):
            print(f"{status}: {count}")
        print("="*60)
    
    def run(self):
        """Main download pipeline"""
        # Create directories
        self.out.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        
        # Load and shuffle image IDs
        all_ids = self.load_image_ids()
        random.Random(self.seed).shuffle(all_ids)
        
        print(f"Target: download {self.target_count} images")
        print(f"Output root: {self.out}")
        print(f"Using {self.num_workers} workers")
        print(f"Retries: {self.max_retries}")
        print(f"Images will be saved to: {self.images_dir}")
        print(f"Log file: {self.log_file}")
        
        # Initialize progress bar
        pbar = tqdm(total=self.target_count, desc="Downloading", unit="img")
        
        # Track retries (but with max_retries=0 by default, this won't be used)
        retry_queue: List[tuple[str, str, int]] = []
        
        # Main download loop
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            idx = 0
            
            # Submit initial batch
            while len(futures) < self.num_workers and idx < len(all_ids):
                split, img_id = all_ids[idx]
                future = executor.submit(self.attempt_download, split, img_id)
                futures[future] = (split, img_id, 0)
                idx += 1
            
            # Process results
            while futures and self.downloaded_count < self.target_count:
                # Wait for next completed download
                done_future = next(as_completed(futures))
                split, img_id, retry_count = futures.pop(done_future)
                
                try:
                    result = done_future.result()
                    self.total_attempted += 1
                    self.results.append(result)
                    self.log_result(result)
                    
                    # Update progress bar for successes
                    if result.status in [DownloadStatus.SUCCESS, DownloadStatus.EXISTS]:
                        pbar.update(1)
                    
                    # Queue for retry if appropriate (only if max_retries > 0)
                    if (self.max_retries > 0 and  # CHANGED: Check if retries are enabled
                        result.status in [DownloadStatus.OTHER_ERROR, DownloadStatus.FORBIDDEN] and 
                        retry_count < self.max_retries):
                        retry_queue.append((split, img_id, retry_count + 1))
                
                except Exception as e:
                    print(f"\nError processing {split}/{img_id}: {e}")
                
                # Submit new tasks
                while len(futures) < self.num_workers and self.downloaded_count < self.target_count:
                    # First try retry queue (only if retries are enabled)
                    if self.max_retries > 0 and retry_queue:  # CHANGED: Check if retries enabled
                        split, img_id, retry_count = retry_queue.pop(0)
                    # Then new items
                    elif idx < len(all_ids):
                        split, img_id = all_ids[idx]
                        retry_count = 0
                        idx += 1
                    else:
                        break
                    
                    future = executor.submit(self.attempt_download, split, img_id, retry_count)
                    futures[future] = (split, img_id, retry_count)
            
            # Cancel remaining futures if target reached
            for future in futures:
                future.cancel()
        
        pbar.close()
        
        # Finalize
        self.write_summary()
        self.print_progress()
        
        # Check if target was reached
        if self.downloaded_count < self.target_count:
            print(f"\nWARNING: Only downloaded {self.downloaded_count}/{self.target_count} images.")
            print("Try increasing the number of IDs or check your internet connection.")
            return False
        
        print(f"\nSUCCESS: Downloaded {self.downloaded_count} images.")
        print(f"   Output root: {self.out}")
        print(f"   Success manifest: {self.success_manifest}")
        print(f"   Detailed log: {self.log_file}")
        print(f"   Summary: {self.summary_file}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenImages with comprehensive logging"
    )
    parser.add_argument(
        "--csv", 
        required=True, 
        type=Path,
        help="Path to image_ids_and_rotation.csv"
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Root directory for output (will create images/ subdirectory)"
    )
    parser.add_argument(
        "--target", 
        type=int, 
        default=10000,
        help="Number of images to download (default: 10000)"
    )
    parser.add_argument(
        "--split", 
        default="train",
        choices=["train", "validation", "test", "challenge2018"],
        help="Which split to download from (default: train)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed for shuffling (default: 0)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=32,
        help="Number of concurrent workers (default: 32)"
    )
    parser.add_argument(
        "--retries", 
        type=int, 
        default=0,
        help="Max retries for failed downloads (default: 0 to avoid blocking)"
    )
    
    args = parser.parse_args()
    
    # Validate CSV exists
    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Run downloader
    downloader = OpenImagesDownloader(
        csv_path=args.csv,
        target_count=args.target,
        out=args.out,
        split=args.split,
        seed=args.seed,
        num_workers=args.workers,
        max_retries=args.retries 
    )
    
    success = downloader.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()