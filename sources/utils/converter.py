import os
import json
import argparse
from tqdm import tqdm
labels = ["motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet", "P2Helmet", "P2NoHelmet", "P0Helmet", "P0NoHelmet"]

parser = argparse.ArgumentParser(description='Convert GT to simpjson')

parser.add_argument('--basepath', type=str, default="data/aicity2024_track5_train/", help='Base path of the dataset')
parser.add_argument('--gtpath', type=str, default="gt.txt", help='Path to the GT file')
parser.add_argument('--outpath', type=str, default="simpjson", help='Path to the output folder')

args = parser.parse_args()

basepath = args.basepath
gtpath = args.gtpath
outpath = args.outpath

outpath = os.path.join(basepath, outpath)
os.makedirs(outpath, exist_ok=True)

with open(os.path.join(basepath, gtpath)) as f:
    c = f.readlines()

video_dict = {}

for l in tqdm(c):
    l = l.strip()
    video_id, frame_id, x, y, w, h, cls_id = l.split(",")
    obj = {'x': x, 'y': y, 'w': w, 'h': h, 'cls_id': cls_id}
    
    if video_id not in video_dict.keys():
        video_dict[video_id] = {frame_id: [obj]}
    else:
        if frame_id not in video_dict[video_id].keys():        
            video_dict[video_id][frame_id] = [obj]
        else:
            video_dict[video_id][frame_id].append(obj)
            
for video_id in tqdm(video_dict.keys()):
    for frame_id in video_dict[video_id]:
#        print(video_id, frame_id, video_dict[video_id][frame_id])
        bboxes = []
        for obj in video_dict[video_id][frame_id]:
            cls_name = labels[int(obj['cls_id'])-1]
#            print(cls_name)
            x1 = obj['x']
            x2 = int(obj['x'])+int(obj['w'])
            y1 = obj['y']
            y2 = int(obj['y'])+int(obj['h'])
            bboxes.append({'class_name': cls_name, 'conf': 1.0, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

#        print(bboxes)
        foutpath = os.path.join(outpath, "%s.mp4_%s.json" %(video_id, frame_id))
        file_name = "%s.mp4_%s.jpg" %(video_id, frame_id)
        simpjson = {"file_name": file_name, "image_width": 0, "image_height": 0, "class_list": [], "bboxes": bboxes, "polygon": [], "file_size": 0, "manual_label": 0, "set_type": ""}
        
        with(open(foutpath, "w")) as f:
            json.dump(simpjson, f)   


    
    
