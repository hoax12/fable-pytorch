"""Regenerate every Tensor Forge notebook: python tools/build_all.py

Point Colab badges at your fork:
    FORGE_GITHUB_REPO=user/repo python tools/build_all.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_act1
import build_act2
import build_act3
import build_onnx_quest


def main():
    print("Forging notebooks...")
    build_act1.build()
    build_act2.build()
    build_act3.build()
    build_onnx_quest.build()
    print("Done — see ./notebooks/")


if __name__ == "__main__":
    main()
