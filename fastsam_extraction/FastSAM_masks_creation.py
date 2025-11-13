import os
import argparse
import json
import numpy as np
from FastSAM import FastSAM, FastSAMPrompt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_split_mapping(scenario_dir):
    split_path = os.path.join(scenario_dir, 'split.json')
    with open(split_path, 'r') as f:
        split_data = json.load(f)

    split_map = {}
    for split, take_list in split_data.items():
        for take_uid in take_list:
            split_map[take_uid] = split
    return split_map

def get_image_paths(root_dir):
    all_data = []
    for scenario in os.listdir(root_dir):
        scenario_path = os.path.join(root_dir, scenario)
        if not os.path.isdir(scenario_path):
            continue

        split_map = load_split_mapping(scenario_path)

        for take_uid in os.listdir(scenario_path):
            take_path = os.path.join(scenario_path, take_uid)
            if not os.path.isdir(take_path):
                continue
            split = split_map.get(take_uid)
            if split is None:
                continue  # skip takes not in split.json

            for cam in os.listdir(take_path):
                cam_path = os.path.join(take_path, cam)
                for fname in os.listdir(cam_path):
                    if fname.endswith(".jpg"):
                        all_data.append({
                            "scenario": scenario,
                            "take_uid": take_uid,
                            "cam": cam,
                            "frame": fname,
                            "image_path": os.path.join(cam_path, fname),
                            "split": split
                        })
    return all_data

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def determine_direction(cam_name):
    cam_lower = cam_name.lower()
    if "cam" in cam_lower:
        return "EXO"
    elif "aria" in cam_lower:
        return "EGO"
    else:
        raise ValueError(f"Cannot determine camera type from name: {cam_name}")

def process_image(model, image_info, weights_root):
    image_path = image_info["image_path"]
    take_uid = image_info["take_uid"]
    cam = image_info["cam"]
    frame_id = os.path.splitext(image_info["frame"])[0]
    split = image_info["split"]

    try:
        results = model(image_path, device=args.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        prompt = FastSAMPrompt(image_path, results, device=args.device)
        annotations = prompt.everything_prompt()

        masks = annotations.masks.data.cpu().numpy()
        boxes = annotations.boxes.data.cpu().numpy()

        # Determine where to save
        cam_type = determine_direction(cam)

        target_dirs = []
        if split == "train":
            # Save both EXO2EGO and EGO2EXO
            target_dirs = [
                f"Masks_TRAIN_EXO2EGO" if cam_type == "EXO" else f"Masks_TRAIN_EGO2EXO",
                f"Masks_TRAIN_EGO2EXO" if cam_type == "EXO" else f"Masks_TRAIN_EXO2EGO",
            ]
        else:
            # Save only source direction
            direction = "EXO2EGO" if cam_type == "EXO" else "EGO2EXO"
            target_dirs = [f"Masks_{split.upper()}_{direction}"]

        for target_dir in target_dirs:
            out_path = os.path.join(weights_root, target_dir, take_uid, cam)
            ensure_dir(out_path)
            np.save(os.path.join(out_path, f"{frame_id}_boxes.npy"), boxes)
            np.savez_compressed(os.path.join(out_path, f"{frame_id}_masks.npz"), masks=masks)
            prompt.plot(annotations=annotations, output_path=os.path.join(out_path, f"{frame_id}_plot.jpg"))

    except Exception as e:
        print(f"Failed processing {image_path}: {e}")

def main(args):
    model = FastSAM(args.weights)
    all_images = get_image_paths(args.image_root)
    print(f"Found {len(all_images)} frames to process.")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_image, model, img_info, args.output_root)
            for img_info in all_images
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=str, default="./output_dir_cooking", help="Path to image root directory")
    parser.add_argument("--weights", type=str, default="./weights/FastSAM.pt", help="Path to FastSAM weights")
    parser.add_argument("--output-root", type=str, default="./Ego-Exo4d", help="Root output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()
    main(args)