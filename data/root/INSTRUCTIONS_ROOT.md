There are two approaches to create the root folder for O-MaMa.

**Approach 1** – we did some work for you:
1. Download the data from the [Google Drive link](https://drive.google.com/drive/folders/1iH0zl6lrLcFze9L25g9JUVA6CBWjOGnJ?usp=drive_link) realted to the Health (in full) and Cooking (in part) scenarios that were explored during the project. Access is subject to confirmation by the authors. The download will be already formatted with the structure that O-MaMa expects. Put this inside the `processed` folder.
2. Run `../../src/scripts/FastSAM_masks_creation.py`, making sure `fastsam_extraction` folder exist in `src` folder. 
3. Run `../../src/scripts/precompute_features_{feature_extractor}.py`, making sure structure that the appropriate strucutre for each feature extractor exists. 

**Approach 2** – start from scratch:
1. Follow the instructions of the [O-MaMa official repository](https://github.com/Maria-SanVil/O-MaMa/tree/main?tab=readme-ov-file). 
2. Run `../../src/scripts/download_and_process_data.py` and rename ending folder as `processed`.
3. Run `../../src/fastsam_extraction/extract_masks_FastSAM.py`, making sure `fastsam_extraction` folder exisst in `src` folder. This will create the `Masks_{split}_{source}2{destination}` folders. 
4. Run `../../src/scripts/precompute_features_{feature_extractor}.py`, making sure structure for appropriate feature extractor folder exists. 

Notes: 
* we focus on not-downscaled data because it contains more information that naturally allows O-MaMa to perform at its best. 
* DINOv2 and ResNet50 are laoded with `torch.load()`; DINOv3 requires cloning its official repository. 


Following either approach, you will obtain this structure inside `root`:

```
root/
├── processed
├── Masks_TRAIN_EXO2EGO
├── Masks_TEST_EXO2EGO
├── Masks_VAL_EXO2EGO
├── dataset_jsons
├── precomputed_features
```

where 
```
└── processed/
│   ├── split.json
│   └── {take_UID_1}/
│       ├── annotation.json
│       └── {cam_1}/
│           └── {frame_1}.jpg
│           └── ...
│       └── {cam_2}/
│           └── {frame_1}.jpg
│           └── ...
```
and
```
├── dataset_jsons/
│   ├── test_egoexo_pairs.json
│   ├── train_egoexo_pairs.json
│   ├── val_egoexo_pairs.json

```
and
```
Masks_TRAIN_EXO2EGO/
├── [take_UID_1]
│    └── [cam]
│          ├── [idx]_boxes.npy
│          └── [idx]_masks.npz
|── ...
Masks_TEST_EXO2EGO/
|── ...
Masks_VAL_EXO2EGO/
|── ...
```
and
```
└── precomputed_features/
│   └── {take_UID_1}/
│       └── {cam_1}/
│           └── {frame_1}.npz
│           └── ...
│       └── {cam_2}/
│           └── {frame_1}.npz
│           └── ...```