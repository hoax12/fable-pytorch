"""The Glyph Oracle — torch-free ONNX inference (companion to Quest 15).

Depends ONLY on numpy + onnxruntime. This is the artifact you'd actually deploy.

Usage (from fable_folder/, after running Quest 15 which exports the artifacts):
    python ship/glyph_oracle.py notebooks/ship/glyphs.onnx --classes notebooks/ship/glyph_classes.json --glyph ring

    # available glyphs: cross, ring, plus, slash
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnxruntime as ort


class GlyphOracle:
    def __init__(self, model_path: str, classes_path: str | None = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{model_path} not found — run notebooks/15_beyond_the_forge_onnx.ipynb to export it."
            )
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.classes = None
        if classes_path and os.path.exists(classes_path):
            with open(classes_path) as f:
                self.classes = json.load(f)

    def __call__(self, image_20x20) -> dict:
        x = np.asarray(image_20x20, dtype=np.float32).reshape(1, 1, 20, 20)
        logits = self.sess.run(None, {self.inp: x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        k = int(p.argmax())
        return {
            "glyph": self.classes[k] if self.classes else str(k),
            "confidence": float(p[k]),
        }


def draw_glyph(name: str, size: int = 20) -> np.ndarray:
    """Numpy re-implementation of the course glyphs (no torch!)."""
    img = np.zeros((size, size), dtype=np.float32)
    ys, xs = np.mgrid[0:size, 0:size]
    c = size // 2
    if name == "cross":
        img[np.abs(xs - ys) <= 1] = 1.0
        img[np.abs(xs + ys - size + 1) <= 1] = 1.0
    elif name == "ring":
        r2 = (xs - c) ** 2 + (ys - c) ** 2
        img[(r2 >= 25) & (r2 <= 49)] = 1.0
    elif name == "plus":
        img[np.abs(ys - c) <= 1] = 1.0
        img[np.abs(xs - c) <= 1] = 1.0
    else:  # slash
        img[np.abs(xs + ys - size + 1) <= 1] = 1.0
    return np.clip(img + 0.08 * np.random.randn(size, size).astype(np.float32), 0, 1)


def main():
    ap = argparse.ArgumentParser(description="Torch-free glyph classification.")
    ap.add_argument("model", help="path to the .onnx artifact")
    ap.add_argument("--classes", default=None, help="path to glyph_classes.json")
    ap.add_argument("--glyph", default="ring", choices=["cross", "ring", "plus", "slash"],
                    help="which synthetic glyph to classify")
    args = ap.parse_args()

    oracle = GlyphOracle(args.model, args.classes)
    result = oracle(draw_glyph(args.glyph))
    print(f"drew a '{args.glyph}' -> oracle says: {result['glyph']} "
          f"({result['confidence'] * 100:.1f}% confident)")


if __name__ == "__main__":
    main()
