import os
import json
import re
import magic
import argparse


def get_img_shape(img_path):
    width = 0
    height = 0
    print(img_path)
    file_magic = magic.from_file(img_path)
    # print("file_magic", file_magic)
    regex_result = re.findall("(\d+)x(\d+)", file_magic)
    # print("regex_result", regex_result)
    if len(regex_result)>1:
        width, height = regex_result[1]
    else:
        width, height = regex_result[0]
    return int(width), int(height)

def create_image_annotation(img_name: str, width: int, height: int, image_id: int):
    image_annotation = {
        "file_name": img_name,
        "height": height,
        "width": width,
        "id": image_id,
    }
    return image_annotation

def create_categories(categories):
    categories_list = []
    index = 1
    for category in categories:
        category_dict = {
            "supercategory": "none",
            "id": index,
            "name": category,
        }
        categories_list.append(category_dict)
        index += 1
    return categories_list

def main(inpath: str, imgpath: str, outpath: str):
    

    labels = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet", "P0Helmet", "P0NoHelmet"]    
    coco_format = {"images": [], "annotations": []}

    annotation_id = 1
    categories = create_categories(labels)
    coco_format["categories"] = categories

    for i, fname in enumerate(os.listdir(inpath)):
        image_id = i+1
        with open(os.path.join(inpath, fname), "r") as fl:
            j = json.load(fl)
        print(j)
        
        width, height = get_img_shape(os.path.join(imgpath, j['file_name']))
        images = create_image_annotation(img_name = j['file_name'], width = width, height = height, image_id = image_id)
        coco_format["images"].append(images)
        
        
        for box in j['bboxes']:
            x1 = int(box['x1'])
            x2 = int(box['x2'])
            y1 = int(box['y1'])
            y2 = int(box['y2'])
            w = x2-x1
            h = y2-y1
            area = w * h
            bbox = [x1, y1, w, h]
            category_id = labels.index(box['class_name'])+1
            seg = []
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
                "category_id": category_id,
                "segmentation": seg,
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1
    #print(coco_format)

    with open(outpath, "w") as f:
        json.dump(coco_format, f)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Convert simple json to coco format')
    parser.add_argument('--basepath', type=str, default="data/aicity2024_track5_train/", help='Base path of the dataset')
    
    args = parser.parse_args()
    basepath = args.basepath
    inpath = os.path.join(basepath, "train_label")
    imgpath = os.path.join(basepath,"train_img")
    outpath = os.path.join(basepath,"train.json")
    main(
        inpath,
        imgpath,
        outpath
    )
