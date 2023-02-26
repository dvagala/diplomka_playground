import json
from lib import *


annotation_id = 1
category_id = 1
coco_dataset_dir = 'coco_dataset'
anotation_masks_dir = 'anotation_masks'


anotation_masks = get_anotated_mask_files_with_corresponding_image_ids(
    anotation_masks_dir, coco_dataset_dir)


annotations = []
for anotation_mask in anotation_masks:
    image_id = anotation_mask[0]
    mask = Image.open(anotation_mask[1])

    print(f'processing image: {image_id}')

    sub_masks = create_sub_masks(mask)
    for color, sub_mask in sub_masks.items():
        annotation = create_sub_mask_annotation(
            sub_mask, image_id, category_id, annotation_id, is_crowd=0)
        if any(annotation["segmentation"]):
            annotations.append(annotation)
            annotation_id += 1


coco_json = {
    "categories": [
            {
                "id": category_id,
                "name": "surface"
            }
    ],
    "images": [{"id": anotation_mask[0], "file_name": anotation_mask[0]} for anotation_mask in anotation_masks],
    "annotations": annotations
}

with open(f'{coco_dataset_dir}/annotations.json', 'w') as f:
    json.dump(coco_json, f, indent=4)
