"""
Search Threshold Optimizer — Automatically finding the "Golden Threshold".
Iterates through similarity scores to find the optimal cut-off for perfect accuracy.
"""

import logging
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.pipeline.face_pipeline import FacePipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def optimize_thresholds(dataset_dir: Path):
    logger.info(f"Optimizing search thresholds using dataset: {dataset_dir}")
    pipeline = FacePipeline()
    
    # 1. Gather all pairs (Same Person vs Different Person)
    identities = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    pos_scores = []
    neg_scores = []
    
    logger.info("Extracting embeddings for calibration...")
    identity_to_embs = {}
    
    for ident_folder in tqdm(identities, desc="Extracting"):
        imgs = list(ident_folder.glob("*"))
        embs = []
        for img_path in imgs[:5]: # Take top 5 images per person
            try:
                pil_img = Image.open(img_path).convert("RGB")
                # We extract embedding directly to bypass retrieval for score gathering
                face_data = pipeline.rcnn_detector.detect_primary(pil_img)
                if face_data:
                    emb = pipeline.embedding.extract(face_data["face_image"])
                    embs.append(emb)
            except: continue
        if len(embs) >= 2:
            identity_to_embs[ident_folder.name] = embs

    # 2. Compute similarity pairs
    all_ident_names = list(identity_to_embs.keys())
    
    for i, name_a in enumerate(all_ident_names):
        embs_a = identity_to_embs[name_a]
        
        # Positive Pairs (Same Identity)
        for j in range(len(embs_a)):
            for k in range(j+1, len(embs_a)):
                sim = np.dot(embs_a[j], embs_a[k])
                pos_scores.append(sim)
        
        # Negative Pairs (Different Identity)
        other_names = all_ident_names[i+1 : i+5] # Sample some other people
        for name_b in other_names:
            embs_b = identity_to_embs[name_b]
            for ea in embs_a:
                for eb in embs_b:
                    sim = np.dot(ea, eb)
                    neg_scores.append(sim)

    # 3. Find Optimal Threshold (Maximize F1 Score)
    best_f1 = 0
    best_threshold = 0.5
    
    thresholds = np.linspace(0.4, 0.95, 56)
    for t in thresholds:
        tp = sum(1 for s in pos_scores if s >= t)
        fp = sum(1 for s in neg_scores if s >= t)
        fn = sum(1 for s in pos_scores if s < t)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    logger.info("\n" + "="*35)
    logger.info("THRESHOLD OPTIMIZATION RESULTS")
    logger.info("="*35)
    logger.info(f"Golden Threshold:    {best_threshold:.3f}")
    logger.info(f"Max F1-Score:        {best_f1:.4f}")
    logger.info(f"Mean Positive Score: {np.mean(pos_scores):.3f}")
    logger.info(f"Mean Negative Score: {np.mean(neg_scores):.3f}")
    logger.info("="*35)
    logger.info(f"\nUpdate settings.SIMILARITY_THRESHOLD to {best_threshold:.3f} for perfect search.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to identity dataset")
    args = parser.parse_args()
    optimize_thresholds(Path(args.dataset))
