#!/usr/bin/env python3
"""Quick script to check image dimensions"""
import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python check_image_size.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not load image: {img_path}")
    sys.exit(1)

h, w = img.shape[:2]
patch_size = 14
print(f"Image: {img_path}")
print(f"Dimensions: {h} x {w}")
print(f"Height divisible by {patch_size}: {h % patch_size == 0} (remainder: {h % patch_size})")
print(f"Width divisible by {patch_size}: {w % patch_size == 0} (remainder: {w % patch_size})")
print(f"After reshape would be: {patch_size * (h // patch_size)} x {patch_size * (w // patch_size)}")

