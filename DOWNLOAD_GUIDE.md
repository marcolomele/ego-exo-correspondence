# Ego4D Download Guide: Filtering by Task IDs and Take UIDs

This guide explains how to download specific data from the Ego4D dataset using task IDs and take UIDs.

## Overview

The `egoexo` CLI doesn't directly support filtering by task ID. Instead, you need to:
1. Extract take UIDs that correspond to your task IDs from the metadata
2. Use `--uids` flag to download only those specific takes

## Step-by-Step Process

### Step 1: Download Metadata First

Before you can filter by task IDs, you need the metadata:

```bash
egoexo -o outdir --parts metadata
```

This downloads `ego4d.json`, `takes.json`, and other metadata files to `outdir/`.

### Step 2: Extract Take UIDs for Your Task IDs

Use the helper script to find take UIDs for specific task IDs:

```bash
# Example: Get all takes for task ID "1001" (Cooking an Omelet)
python get_take_uids_by_task.py outdir/metadata 1001

# Multiple task IDs
python get_take_uids_by_task.py outdir/metadata 1001 1002 1003

# Task IDs from your metadata.json (e.g., "5001" for Playing Guitar)
python get_take_uids_by_task.py outdir/metadata 5001
```

This will output space-separated take UIDs that you can use for download.

### Step 3: Download Data for Specific Takes

Use the `--uids` flag with the extracted take UIDs:

```bash
# Download specific takes (replace with actual UIDs from step 2)
egoexo -o ego4d_data \
  --parts takes take_vrs take_trajectory take_eye_gaze \
  --uids <uid1> <uid2> <uid3> \
  -y

# Or save UIDs to a file and use them
TAKE_UIDS=$(python get_take_uids_by_task.py outdir/metadata 1001)
egoexo -o ego4d_data --parts takes --uids $TAKE_UIDS -y
```

## Alternative: Filter by Benchmark

If you're working with the correspondence benchmark, you can filter directly:

```bash
egoexo -o ego4d_data \
  --parts takes take_vrs annotations \
  --benchmarks correspondence \
  --splits train \
  -y
```

This downloads all takes that are part of the correspondence benchmark in the training split.

## Complete Example Workflow

```bash
# 1. Download metadata
egoexo -o outdir --parts metadata -y

# 2. Get take UIDs for cooking tasks (1001, 1002, etc.)
TAKE_UIDS=$(python get_take_uids_by_task.py outdir/metadata 1001 1002 1003)
echo "Found take UIDs: $TAKE_UIDS"

# 3. Download data for those takes
egoexo -o ego4d_data \
  --parts takes take_vrs take_trajectory \
  --uids $TAKE_UIDS \
  -y
```

## Task ID Reference

From your `metadata.json`, common task IDs include:

- **Cooking (1000-1018)**: e.g., "1001" = Cooking an Omelet, "1008" = Cooking Sushi Rolls
- **Health (2000-2005)**: e.g., "2001" = Covid-19 Rapid Antigen Test
- **Music (5000-5021)**: e.g., "5001" = Playing Guitar, "5009" = Playing Piano
- **Basketball (6000-6003)**: e.g., "6001" = Basketball Drills - Mikan Layup
- **Soccer (8000-8005)**: e.g., "8001" = Soccer Drills - Inside Trap
- **Dance (9000-9007)**: e.g., "9001" = Performing the basic choreography

## View Available Options

Check all available filters:

```bash
egoexo --help
```

Key flags:
- `--uids`: Filter by take/capture UIDs (what you need for specific takes)
- `--benchmarks`: Filter by benchmark name (correspondence, relations, etc.)
- `--splits`: Filter by train/val/test
- `--views`: Filter by ego/exo
- `--parts`: What data to download (takes, take_vrs, annotations, etc.)

## Troubleshooting

If the script doesn't find takes:
1. Verify metadata was downloaded: `ls outdir/metadata/`
2. Check if `takes.json` exists: `ls outdir/metadata/takes.json` or `ls outdir/metadata/v2/takes.json`
3. Verify task IDs exist in metadata: Check `metadata.json` for valid task IDs
4. Task IDs might be stored differently - check the structure of `takes.json` manually

