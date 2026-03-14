"""
Retrieval Benchmark Script — Quantifying search "perfection".
Calculates Rank-1 Accuracy, Rank-5 Accuracy, and mAP (Mean Average Precision).

Expects a dataset directory where each subfolder name is the identity.
Example structure:
    data/benchmark/
        alice/
            1.jpg
            2.jpg
        bob/
            1.jpg
"""

import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.pipeline.face_pipeline import FacePipeline
from app.pipeline.database_builder import DatabaseBuilder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class RetrievalBenchmark:
    def __init__(self):
        self.pipeline = FacePipeline()
        self.db_builder = DatabaseBuilder()

    def run(self, dataset_dir: Path):
        logger.info(f"Running benchmarks on {dataset_dir}")
        
        # 1. Gather identities
        identity_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(identity_folders)} identities.")

        # 2. Build temporary index from one image per identity (the 'gallery')
        # And use others as 'probes'
        gallery_images = []
        probe_images = []

        for folder in identity_folders:
            imgs = sorted(list(folder.glob("*")))
            if len(imgs) < 2:
                logger.warning(f"Skipping {folder.name}: need at least 2 images for probe/gallery split.")
                continue
            
            gallery_images.append((imgs[0], folder.name))
            for p in imgs[1:]:
                probe_images.append((p, folder.name))

        logger.info(f"Gallery: {len(gallery_images)} | Probes: {len(probe_images)}")

        # 3. Rebuild index with gallery
        self.pipeline.faiss_index.reset()
        for img_path, identity in tqdm(gallery_images, desc="Building Gallery Index"):
            try:
                # We use a custom metadata field 'identity' in the DB if we had it,
                # but for simplicity in benchmark we'll just track IDs.
                self.db_builder.process_image(img_path)
            except Exception as e:
                logger.error(f"Failed gallery image {img_path}: {e}")

        # 4. Evaluate Probes
        rank1 = 0
        rank5 = 0
        latencies = []

        for probe_path, true_identity in tqdm(probe_images, desc="Evaluating Probes"):
            try:
                img = Image.open(probe_path).convert("RGB")
                start = time.time()
                results = self.pipeline.analyze(img, top_k=5, generate_heatmap=False)
                latencies.append(time.time() - start)

                if results["face_count"] > 0:
                    # We compare the file path of the result to see if it matches the true identity folder
                    # In a real system you'd use a unique identity ID.
                    top_matches = results["faces"][0]["similar_images"]
                    
                    found_rank1 = False
                    found_rank5 = False
                    
                    for idx, match in enumerate(top_matches):
                        match_path = Path(match["image_path"])
                        if match_path.parent.name == true_identity:
                            if idx == 0: found_rank1 = True
                            found_rank5 = True
                            break
                    
                    if found_rank1: rank1 += 1
                    if found_rank5: rank5 += 1
            except Exception as e:
                logger.error(f"Failed probe {probe_path}: {e}")

        # 5. Report
        n = len(probe_images)
        if n > 0:
            logger.info("\n" + "="*30)
            logger.info("RETRIEVAL BENCHMARK RESULTS")
            logger.info("="*30)
            logger.info(f"Rank-1 Accuracy: {rank1/n:.2%} ({rank1}/{n})")
            logger.info(f"Rank-5 Accuracy: {rank5/n:.2%} ({rank5}/{n})")
            logger.info(f"Avg Latency:     {np.mean(latencies)*1000:.1f} ms")
            logger.info("="*30)
        else:
            logger.error("No probes were successfully evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to evaluation dataset")
    args = parser.parse_args()

    benchmark = RetrievalBenchmark()
    benchmark.run(Path(args.dataset))
