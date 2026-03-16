#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Download and process original images."""

import json
import hashlib
import io
import subprocess
import math
import os
from datasets import load_dataset
from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image as PILImage
from tqdm import tqdm
import shutil
from zipfile import ZipFile
import urllib.request

DATASET_NAMES = [
    "omnidocbench",
    "hiertext",
    "rico",
    "screenspotpro",
    "webui",
    "omnidocbench",
    "docvqa",
    "infographicvqa",
    "chartmuseum",
    "chartqapro",
    "vistext",
]


def process_hiertext(output_dir: str) -> None:
    """Download HierText OCR test split from Open Images S3 and organize files.

    Original data stored at: https://github.com/google-research-datasets/hiertext?tab=readme-ov-file

    Args:
        output_dir: Base directory to save processed data.
    """
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    hiertext_dir = base_dir / "hiertext"
    hiertext_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = hiertext_dir / "test.tgz"
    images_dir = hiertext_dir / "images"

    print("Downloading HierText test.tgz from S3...")
    subprocess.run(
        [
            "aws",
            "s3",
            "--no-sign-request",
            "cp",
            "s3://open-images-dataset/ocr/test.tgz",
            str(tgz_path),
        ],
        check=True,
    )

    print("Extracting test.tgz...")
    subprocess.run(
        ["tar", "-xzvf", str(tgz_path)],
        cwd=hiertext_dir,
        check=True,
    )

    print("Creating target directory...")
    images_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = hiertext_dir / "test"
    if not extracted_dir.exists():
        raise RuntimeError("Expected 'test/' directory not found after extraction")

    shutil.move(str(extracted_dir), images_dir)
    print(f"HierText images saved to: {images_dir}")


def process_rico(output_dir: str) -> None:
    """Download rico dataset and organize files.

    Original data stored at: https://www.interactionmining.org/archive/rico

    Args:
        output_dir: Base directory to save processed data.
    """

    base_dir = Path(output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    rico_dir = base_dir / "rico"
    rico_dir.mkdir(parents=True, exist_ok=True)

    tar_path = rico_dir / "unique_uis.tar.gz"
    images_dir = rico_dir / "images"

    url = (
        "https://storage.googleapis.com/"
        "crowdstf-rico-uiuc-4540/"
        "rico_dataset_v0.1/"
        "unique_uis.tar.gz"
    )

    print("Downloading RICO dataset...")
    urllib.request.urlretrieve(url, tar_path)

    print("Extracting archive...")
    subprocess.run(
        ["tar", "-xzvf", str(tar_path)],
        cwd=rico_dir,  # Should be saved to rico_dir/combined folder.
        check=True,
    )

    extracted_dir = rico_dir / "combined"
    if not extracted_dir.exists():
        raise RuntimeError("Expected 'combined/' directory not found after extraction")

    print("Creating target directory...")
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Moving images...")
    for item in extracted_dir.iterdir():
        shutil.move(str(item), images_dir / item.name)

    print("Cleaning up intermediate files...")
    if tar_path.exists():
        tar_path.unlink()
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    print("\nCompleted!")
    print(f"RICO images saved to: {images_dir}")


def process_omnidocbench(output_dir: str, exclude_cn: bool = True) -> None:
    """Download OmniDocBench dataset and organize images.

    Args:
        output_dir: Base directory to save processed data.
        exclude_cn: If True, exclude Chinese and mixed language samples.
    """
    base_dir = Path(output_dir)
    images_dir = base_dir / "omnidocbench" / "images"
    metadata_path = base_dir / "omnidocbench" / "OmniDocBench.json"
    images_dir.mkdir(parents=True, exist_ok=True)
    local_repo_dir = None

    print("Downloading OmniDocBench dataset from Hugging Face...")
    repo_id = "opendatalab/OmniDocBench"
    local_repo_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    print(f"Dataset downloaded to: {local_repo_dir}")

    if metadata_path.exists() or (
        local_repo_dir and (Path(local_repo_dir) / "OmniDocBench.json").exists()
    ):
        if local_repo_dir:
            downloaded_metadata_path = Path(local_repo_dir) / "OmniDocBench.json"
            if downloaded_metadata_path.exists():
                metadata_path = downloaded_metadata_path
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if exclude_cn:
            print("Filtering out Chinese and mixed language samples...")
            filtered_metadata = []
            excluded_count = 0
            for entry in metadata:
                if "page_info" in entry and "page_attribute" in entry["page_info"]:
                    page_attr = entry["page_info"]["page_attribute"]
                    if "language" in page_attr:
                        language = page_attr["language"].lower()
                        if "chinese" in language or "en_ch_mixed" in language:
                            excluded_count += 1
                            continue
                filtered_metadata.append(entry)
            print(
                f"Excluded {excluded_count} Chinese/mixed samples, keeping {len(filtered_metadata)} samples"
            )
            metadata = filtered_metadata
        src_images = Path(local_repo_dir) / "images"
        for idx, entry in enumerate(tqdm(metadata, desc="Processing images")):
            image_id = f"{idx:06d}"
            if "page_info" in entry and "image_path" in entry["page_info"]:
                image_name = entry["page_info"]["image_path"]
                img_path = src_images / image_name
                try:
                    img = PILImage.open(img_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    output_path = images_dir / f"{image_id}.jpg"
                    img.save(output_path, "JPEG", quality=95)
                except Exception as e:
                    print(
                        f"Warning: Failed to process image {image_name} for {image_id}: {e}"
                    )
    num_images = len(list(images_dir.glob("*.jpg")))
    print("\nCompleted!")
    print(f"Images saved: {num_images} files in {images_dir}")


def process_docvqa_infographicvqa(output_dir: str) -> None:
    """Download DocVQA and InfographicVQA dataset organize images.

    Args:
        output_dir: Base directory to save processed data.
    """

    def _download_and_save_images(
        dataset: str,
        sub_dataset: str,
        output_dir: str,
        split_name: str = "test",
    ) -> None:
        """Downloads dataset and save all images to a directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        ds = load_dataset(dataset, sub_dataset)
        print("Dataset loaded successfully. Available splits: %s", list(ds.keys()))

        total_images_saved = 0
        seen_hashes = set()

        assert split_name in ds

        split_data = ds[split_name]
        print("Processing split '%s' with %d samples", split_name, len(split_data))

        img_dir = output_path / sub_dataset / "images"
        img_dir.mkdir(exist_ok=True, parents=True)

        images_in_split = 0
        for idx, sample in enumerate(split_data):
            try:
                image = sample.get("image")  # noqa: E1101
                if image is None:
                    print(
                        "No image found in sample %d of split '%s'",
                        idx,
                        split_name,
                    )
                    continue

                img_byte_arr = io.BytesIO()
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_byte_arr, format="JPEG", quality=95)
                image_hash = hashlib.sha256(img_byte_arr.getvalue()).hexdigest()

                if image_hash in seen_hashes:
                    continue
                seen_hashes.add(image_hash)

                question_id = sample.get(  # noqa: E1101
                    "questionId",
                    sample.get("question_id", f"{split_name}_{idx}"),  # noqa: E1101
                )
                filename = f"{question_id}.jpg"
                image_path = img_dir / filename

                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, "JPEG", quality=95)
                images_in_split += 1
                total_images_saved += 1

            except Exception as e:
                print(
                    "Error processing sample %d in split '%s': %s",
                    idx,
                    split_name,
                    e,
                )
                continue

        print("Completed split '%s': saved %d images", split_name, images_in_split)

        print("Download completed! Total images saved: %d", total_images_saved)
        print("Images saved to: %s", img_dir.absolute())

    _download_and_save_images(
        dataset="lmms-lab/DocVQA",
        sub_dataset="DocVQA",
        output_dir=output_dir,
        split_name="test",
    )

    _download_and_save_images(
        dataset="lmms-lab/DocVQA",
        sub_dataset="InfographicVQA",
        output_dir=output_dir,
        split_name="test",
    )

    output_path = Path(output_dir)
    os.rename(output_path / "InfographicVQA", output_path / "infographicvqa")
    os.rename(output_path / "DocVQA", output_path / "docvqa")


def process_chartmuseum_chartqapro(output_dir: str) -> None:
    """Download ChartMuseum and ChartQAPro datasets and organize images.

    Args:
        output_dir: Base directory to save processed data.
    """
    base_dir = Path(output_dir)

    # Download ChartMuseum using snapshot_download
    print("Downloading ChartMuseum dataset...")
    chartmuseum_dir = base_dir / "chartmuseum"

    local_repo = snapshot_download(
        repo_id="lytang/ChartMuseum",
        repo_type="dataset",
        local_dir=str(chartmuseum_dir),
        allow_patterns=["images/*"],
    )
    print(f"ChartMuseum downloaded to: {local_repo}")

    # Download ChartQAPro using load_dataset (images embedded in parquet as binary)
    print("\nDownloading ChartQAPro dataset...")
    chartqapro_dir = base_dir / "chartqapro"
    images_dir = chartqapro_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset - this reads the parquet with embedded images
    ds = load_dataset("ahmed-masry/ChartQAPro")

    # Extract and save images from the 'test' split
    print(f"Extracting {len(ds['test'])} images from ChartQAPro...")
    for idx, sample in enumerate(tqdm(ds["test"], desc="Extracting images")):
        image_data = sample["image"]  # type: ignore

        # Handle both PIL Image and bytes formats
        if isinstance(image_data, bytes):
            image = PILImage.open(io.BytesIO(image_data))
        else:
            # Already a PIL Image (datasets library auto-decodes)
            image = image_data

        if image.mode != "RGB":
            image = image.convert("RGB")

        filename = f"chart_{idx:04d}.jpg"
        image.save(images_dir / filename, "JPEG", quality=95)

    print(f"ChartQAPro images saved to: {images_dir}")
    print("\nAll datasets downloaded successfully!")


def get_optimal_grid_layout(n_images: int) -> tuple[int, int]:
    """Calculate optimal grid layout for n images.

    Args:
        n_images: Number of images to arrange

    Returns:
        Tuple of (rows, cols)
    """
    if n_images == 1:
        return (1, 1)
    elif n_images == 2:
        return (1, 2)
    elif n_images == 3:
        return (1, 3)
    elif n_images == 4:
        return (2, 2)
    elif n_images == 5:
        return (2, 3)
    elif n_images == 6:
        return (2, 3)
    else:
        # For larger numbers, try to make it roughly square
        cols = math.ceil(math.sqrt(n_images))
        rows = math.ceil(n_images / cols)
        return (rows, cols)


def resize_image_proportional(
    image: PILImage.Image, target_width: int, target_height: int
) -> PILImage.Image:
    """Resize image while maintaining aspect ratio to fit within target dimensions.

    Args:
        image: PIL Image to resize
        target_width: Maximum width
        target_height: Maximum height

    Returns:
        Resized PIL Image with maximum quality
    """
    original_width, original_height = image.size

    # Calculate scaling factor to fit within target dimensions
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height
    scale_factor = min(width_ratio, height_ratio)

    # Only resize if the image is larger than target
    if scale_factor >= 1.0:
        return image  # Keep original size if it's already smaller

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Use LANCZOS for high-quality downsampling
    return image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)


def merge_cluster_images_vistext(
    cluster_files: list[str],
    images_dir: str,
    output_path: str,
    max_image_size: tuple[int, int] = (800, 600),
) -> str | None:
    """Merge images from a cluster into a single figure.

    Args:
        cluster_files: List of image filenames in the cluster
        images_dir: Directory containing the images
        output_path: Path to save the merged image
        max_image_size: Maximum size for each individual image in the grid

    Returns:
        Filename of the merged image or None if failed
    """
    # Load all images
    images = []
    valid_files = []

    for image_name in cluster_files:
        image_path = os.path.join(images_dir, image_name)

        try:
            img = PILImage.open(image_path)
            # Convert to RGB if necessary (in case of RGBA or other modes)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
            valid_files.append(image_name)
        except (FileNotFoundError, IOError) as e:
            print(f"Warning: Could not load image {image_path}: {e}")

    if not images:
        print("Warning: No valid images found for cluster")
        return None

    # Calculate grid layout
    rows, cols = get_optimal_grid_layout(len(images))

    # Resize all images to fit within max_image_size while maintaining aspect ratio
    resized_images = []
    for img in images:
        resized_img = resize_image_proportional(
            img, max_image_size[0], max_image_size[1]
        )
        resized_images.append(resized_img)

    # Calculate the size of the merged image
    # Use the maximum dimensions among resized images for consistent grid
    max_width = max(img.width for img in resized_images)
    max_height = max(img.height for img in resized_images)

    # Add padding between images
    padding = 20
    merged_width = cols * max_width + (cols + 1) * padding
    merged_height = rows * max_height + (rows + 1) * padding

    # Create the merged image with white background
    merged_image = PILImage.new("RGB", (merged_width, merged_height), "white")

    # Place images in the grid
    for i, img in enumerate(resized_images):
        row = i // cols
        col = i % cols

        # Calculate position (center the image in its grid cell)
        x = col * (max_width + padding) + padding + (max_width - img.width) // 2
        y = row * (max_height + padding) + padding + (max_height - img.height) // 2

        merged_image.paste(img, (x, y))

    # Save the merged image with maximum quality
    merged_image.save(output_path, "PNG", optimize=False, compress_level=1)
    return os.path.basename(output_path)


def process_vistext(output_dir: str, force_download: bool = False) -> None:
    """Download and process vistext dataset by merging cluster images.

    Args:
        output_dir: Base directory to save processed data
        force_download: If True, download even if data exists
    """
    base_dir = Path(output_dir)
    merged_images_dir = base_dir / "vistext" / "images"
    merged_images_dir.mkdir(parents=True, exist_ok=True)

    vistext_data_dir = base_dir / "vistext_download"
    vistext_data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = vistext_data_dir / "images"

    # Download dataset if needed
    if force_download or not images_dir.exists():
        print(f"Downloading vistext dataset to {vistext_data_dir}...")
        try:
            subprocess.run(
                [
                    "wget",
                    "https://vis.csail.mit.edu/vistext/images.zip",
                    "-P",
                    str(vistext_data_dir),
                ],
                check=True,
            )
            print("Download complete. Extracting...")
            subprocess.run(
                [
                    "unzip",
                    "-q",
                    str(vistext_data_dir / "images.zip"),
                    "-d",
                    str(vistext_data_dir),
                ],
                check=True,
            )
            print("Extraction complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading/extracting dataset: {e}")
            return
    else:
        print(f"Using existing vistext data at {vistext_data_dir}")

    # Check that images directory exists
    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        return

    # Load vistext_mapping.json
    mapping_file = Path(__file__).parent / "data" / "vistext_mapping.json"
    if not mapping_file.exists():
        print(f"Error: vistext_mapping.json not found at {mapping_file}")
        return

    with open(mapping_file, "r") as f:
        vistext_mapping = json.load(f)

    print(f"Processing {len(vistext_mapping)} clusters...")

    # Process each cluster
    for cluster_name, image_files in tqdm(
        vistext_mapping.items(), desc="Merging cluster images"
    ):
        output_path = merged_images_dir / cluster_name

        try:
            result = merge_cluster_images_vistext(
                image_files, str(images_dir), str(output_path)
            )
            if result:
                print(
                    f"Created merged image: {cluster_name} from {len(image_files)} images"
                )
        except Exception as e:
            print(f"Error creating merged image for {cluster_name}: {e}")

    num_merged = len(list(merged_images_dir.glob("*.png")))
    print("\nCompleted!")
    print(f"Merged images saved: {num_merged} files in {merged_images_dir}")


def process_screenspotpro(output_dir: str) -> None:
    """Download ScreenSpot Pro dataset and organize images.

    Args:
        output_dir: Base directory to save processed data.
    """
    base_dir = Path(output_dir)
    images_dir = base_dir / "screenspotpro"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ScreenSpot Pro dataset from Hugging Face...")
    snapshot_download(
        repo_id="likaixin/ScreenSpot-Pro",
        repo_type="dataset",
        local_dir=images_dir,
        allow_patterns=["images/*"],
        resume_download=True,
    )

    print("\nCompleted!")
    print(f"Images saved in {images_dir}")


def process_webui(output_dir: str) -> None:
    """Download webui dataset and organize images.

    Args:
        output_dir: Base directory to save processed data.
    """
    base_dir = Path(output_dir)
    print("Downloading WebUI test dataset...")
    webui_dir = base_dir / "webui"

    snapshot_download(
        repo_id="biglab/webui-test",
        repo_type="dataset",
        local_dir=str(webui_dir),
    )

    part_files = sorted(webui_dir.glob("test_split_webui.zip.*"))
    if not part_files:
        raise FileNotFoundError("No split zip parts found in " + str(webui_dir))

    print(f"Found {len(part_files)} parts; concatenating…")

    combined_zip_path = webui_dir / "webui_combined.zip"
    with combined_zip_path.open("wb") as out:
        for part in part_files:
            print(" -> appending", part.name)
            with part.open("rb") as partfile:
                shutil.copyfileobj(partfile, out)

    extract_dir = webui_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting to", extract_dir)

    with ZipFile(combined_zip_path, "r") as z:
        z.extractall(extract_dir)


def process_raw_datasets(output_dir: str) -> None:
    """Process raw datasets and save them into `output_dir`.

    Args:
        output_dir: Base directory to save processed data.
    """
    process_hiertext(output_dir)
    process_rico(output_dir)
    process_omnidocbench(output_dir)
    process_docvqa_infographicvqa(output_dir)
    process_chartmuseum_chartqapro(output_dir)
    process_vistext(output_dir)
    process_screenspotpro(output_dir)
    process_webui(output_dir)

    # Check that dataset is built corrected.
    base_dir = Path(output_dir)
    for dataset_name in DATASET_NAMES:
        dataset_dir = base_dir / dataset_name
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} is not created correctly.")


if __name__ == "__main__":
    process_raw_datasets("data/")
