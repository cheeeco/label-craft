#!/usr/bin/env python3
"""
Download data files from URLs specified in the config file using Hydra.
"""

import time
from pathlib import Path

import requests
from hydra import compose, initialize
from loguru import logger


def main() -> None:
    """Main function to download all dataset files."""
    try:
        # Initialize Hydra and compose config without creating outputs folder
        with initialize(config_path="../../config", version_base=None):
            cfg = compose(config_name="data_download")

        # Get output directory and download config
        output_dir = Path(cfg.output_dir)
        download_config = cfg.data.download
        dataset_files = cfg.data.datasets.files

        logger.info(f"Starting download to {output_dir}")

        # Download all files
        success_count = 0
        total_count = len(dataset_files)

        for _, file in dataset_files.items():
            url = file.url
            file_name = file.filename
            if not url:
                logger.warning(f"No URL found for {file_name}")
                continue

            # Create output path
            output_path = output_dir / file_name

            # Check if file already exists
            if output_path.exists():
                logger.info(f"File {output_path} already exists, skipping download")
                success_count += 1
                continue

            # Download the file
            if download_file(
                url=url,
                output_path=output_path,
                chunk_size=download_config.chunk_size,
                timeout=download_config.timeout,
                max_retries=download_config.retry_attempts,
            ):
                success_count += 1

        # Summary
        if success_count == total_count:
            logger.success(f"Successfully downloaded all {total_count} files")
        else:
            logger.warning(f"Downloaded {success_count}/{total_count} files")

    except Exception as e:
        logger.error(f"Fatal error in download process: {e}")
        raise


def download_file(url, output_path, chunk_size=8192, timeout=30, max_retries=3):
    """Download a file with retry mechanism."""
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")

            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            # Create parent directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.debug(
                                f"Downloaded {percent:.1f}% of " f"{output_path.name}"
                            )

            logger.success(f"Successfully downloaded {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False


if __name__ == "__main__":
    main()
