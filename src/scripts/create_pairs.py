import json
import argparse

def make_pairs(data_dir, scenario, split='train', setting='egoexo'):
    
    with open(f'{data_dir}/split.json', 'r') as fp:
        splits = json.load(fp)

    split_takes = splits[split]

    pairs = []
    for take in split_takes:

        with open(f'{data_dir}/{take}/annotation.json', 'r') as fp:
            annotation = json.load(fp) 

        for obj_name in annotation['masks']:

            ego_cam = None
            for cam in annotation['masks'][obj_name]:
                if 'aria' in cam:
                    ego_cam = cam

            if ego_cam is None:
                continue

            for cam in annotation['masks'][obj_name]:
                
                if 'aria' in cam:
                    continue

                if setting == 'egoexo':
                    dcam = ego_cam
                elif setting == 'exoego':
                    dcam = cam
                else:
                    raise Exception(f"Setting {setting} not recognized.")

                for idx in annotation['masks'][obj_name][dcam]:
                    cam_rgb_path = f'{data_dir}//{take}//{cam}//{obj_name}//rgb//{idx}'
                    cam_mask_path = f'{data_dir}//{take}//{cam}//{obj_name}//mask//{idx}'

                    aria_rgb_path = f'{data_dir}//{take}//{ego_cam}//{obj_name}//rgb//{idx}'
                    aria_mask_path = f'{data_dir}//{take}//{ego_cam}//{obj_name}//mask//{idx}'

                    pairs.append((aria_rgb_path, aria_mask_path, cam_rgb_path, cam_mask_path))

    print(f'{scenario} {split} {setting} pairs: {len(pairs)}')
    with open(f'{data_dir}/{split}_{setting}_pairs.json', 'w') as fp:
        json.dump(pairs, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True, choices=['cooking', 'health'],
                        help='Scenario to process (cooking or health)')
    args = parser.parse_args()
    
    data_dir = f'output_dir_{args.scenario}'
    
    # Load splits to see what's available
    with open(f'{data_dir}/split.json', 'r') as fp:
        available_splits = json.load(fp)
    
    print(f"Available splits: {list(available_splits.keys())}")
    total_num_takes = sum(len(v) for v in available_splits.values())
    print(f"Total number of takes: {total_num_takes}")
    
    # Process all available splits for both settings
    for split in available_splits.keys():
        make_pairs(data_dir, args.scenario, split, 'exoego')