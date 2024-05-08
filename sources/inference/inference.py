# CUDA_VISIBLE_DEVICES=1 python3 inference.py --test_path /mlcv1/WorkingSpace/Personal/haov/aicity2023/Track5_2024/aicity2024_track5_train/test-set/aicity2024_track5_test/videos/
from mmdet.apis import init_detector, inference_detector
from mmdet.core import DatasetEnum
import mmcv
import cv2
import numpy as np
import os
import argparse

from ensemble_boxes import *
from tqdm import tqdm

from utils.filter2 import Filter
from utils.detection_object import Human, Motor

def process_objects(vid, fid, human_list, motor_list):
    filter = Filter(motor_list, human_list)
    result = ''
    all_class = filter.create_virtual()
    for obj in all_class:
        left, top, right, bottom, class_id, conf, _ = obj.get_box_info()
        result += ','.join(map(str, [vid, fid, left, top, right - left, bottom - top, class_id, conf])) + '\n'
    return result

def process_video(dataset, vid):
    result = ''
    for fid in dataset[vid].keys():
        if 'human' not in dataset[vid][fid].keys():
            dataset[vid][fid]['human'] = []
        if 'motor' not in dataset[vid][fid].keys():
            dataset[vid][fid]['motor'] = []
        result += process_objects(vid, fid, dataset[vid][fid]['human'], dataset[vid][fid]['motor'])
    return result

def Virtural_Expander(data: list):
    dataset = {}
    for line in data:
        vid, fid, left, top, width, height, cls, conf = line
        if int(float(cls)) != 1:
            
            if vid not in dataset.keys():
                dataset[vid] = {}
            if fid not in dataset[vid].keys():
                dataset[vid][fid] = {}
            if 'human' not in dataset[vid][fid].keys():
                dataset[vid][fid]['human'] = []
            dataset[vid][fid]['human'].append(Human(bbox=[float(left), float(top), float(width), float(height),float(cls), float(conf)]))
       
        else:
            if vid not in dataset.keys():
                dataset[vid] = {}
            if fid not in dataset[vid].keys():
                dataset[vid][fid] = {}
            if 'motor' not in dataset[vid][fid].keys():
                dataset[vid][fid]['motor'] = []
            dataset[vid][fid]['motor'].append(Motor(bbox=[float(left), float(top), float(width), float(height),float(cls), float(conf)]))
       
            # if 'human' not in dataset[vid][fid].keys():
            #     dataset[vid][fid]['human'] = []
            # dataset[vid][fid]['human'].append(Human(bbox=[float(left), float(top), float(width), float(height),float(cls), float(conf)]))
    # Create ouput
    results = ''
    for vid in tqdm(dataset.keys()):
        results += process_video(dataset, vid)
    return results

def count_samples_per_class(data):
    class_counts = [0,0,0,0,0,0,0,0,0] 
    for line in data:
        class_id = int(line[-2]) 
        class_counts[class_id-1] += 1
    return class_counts
def find_max(classes):
    classes_count = count_samples_per_class(classes)
    max_class = max(classes_count)
    return max_class, classes_count


def minority(p, classes):
    n_maxclass, classes_count = find_max(classes)
    mean_samples = float(len(classes)/9)
    alpha = mean_samples/n_maxclass
    rare_classes = []
    for index, each_class in enumerate(classes_count):
        n_class = each_class
        if n_class < (n_maxclass * alpha):
            rare_classes.append(index)
    min_thresh = 1
    for each_class_index in rare_classes:
        for each_sample in classes:
            if each_class_index != int(each_sample[-2]-1):
                continue
            if each_sample[-1] < min_thresh:
                min_thresh = each_sample[-1]
    return max(min_thresh, p)

def read_detections(lines: list):
    detections_dict = {}
    w, h = 1920, 1080 # NOTE: Change this to the actual width and height of the video
    for line in lines:
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, score = line.strip().split(',')
        frame = int(float(frame))
        video_id = int(video_id)
        if video_id not in detections_dict:
            detections_dict[video_id] = {}
        if frame not in detections_dict[video_id]:
            detections_dict[video_id][frame] = []
        detections_dict[video_id][frame].append([float(bb_left) / w, float(bb_top) / h, (float(bb_width) + float(bb_left)) / w, 
        (float(bb_height) + float(bb_top)) / h,  float(score), int(float(class_id)), ])
    return detections_dict

def detect_video(
    test_path: str,
    config_path: str,
    checkpoint_files: list,
    batch_size: int,
) -> list:
    process_video_results = []
    configs_weights = [
        ('co_dino_5scale_swin_large_16e_o365tococo.py','epoch_10.pth'),
        ('640x640co_dino_5scale_swin_large_16e_o365tococo.py','epoch_10.pth'),
        ('1280x1280co_dino_5scale_swin_large_16e_o365tococo.py','epoch_10.pth'),
        ('640x640co_dino_5scale_swin_large_16e_o365tococo.py','epoch_15.pth'),
        ('1280x1280co_dino_5scale_swin_large_16e_o365tococo.py','epoch_15.pth'),
    ]
    for config_name, checkpoint_file in configs_weights:
        config_f_name = config_name.split(".")[0]
        checkpoint_file = os.path.join(checkpoint_files, checkpoint_file)

        lines = []
        config_file = os.path.join(config_path, config_name)
        model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:0')

        for video_name in tqdm(os.listdir(test_path)):
            video_id = video_name.split(".")[0]
            video_path = os.path.join(test_path, video_name)
            frame_id = 0
            cap = cv2.VideoCapture(video_path)
            batch = []
            is_break = False
            while True:
                while len(batch) < batch_size:
                    ret, img = cap.read()
                    if not ret:
                        is_break = True
                        break
                    batch.append(img)
                if is_break:
                    break
                print(f"[INFO] Current frame_id: {frame_id}")
                results = inference_detector(model, batch)
                for idx, result in enumerate(results):
                    bbox_result, segm_result = result, None
                    bboxes = np.vstack(bbox_result)
                    labels = [
                        np.full(bbox.shape[0], i, dtype=np.int32)
                        for i, bbox in enumerate(bbox_result)
                    ]
                    labels = np.concatenate(labels)
                    score_thr = 0.01
                    scores = None
                    if score_thr > 0:
                        scores = bboxes[:, -1]
                        inds = scores > score_thr
                        scores = scores[inds]
                        bboxes = bboxes[inds, :]
                        labels = labels[inds]
                    width, height = img.shape[1], img.shape[0]
                    for label, score, bbox in zip(labels, scores, bboxes):
                        bbox = list(map(int, bbox))
                        label = int(label) + 1
                        w,h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        lines.append(
                            f"{int(video_id)},{frame_id + idx + 1},{bbox[0]},{bbox[1]},{w},{h},{label},{score}\n"
                        )
                frame_id += len(batch)
                batch = []
            process_video_results.append(lines)
        
    return process_video_results


def fuse(
    process_video_results: list,
    video_path: str,
    iou_thr: float = 0.7, # default values of repo
    skip_box_thr: float = 0.0001, # default values of repo
) -> list:
    datas = [read_detections(item) for item in process_video_results]
    results = []
    w, h = 1920, 1080

    for video_name in tqdm(os.listdir(video_path)):
        video_id = int(video_name.split(".")[0])
        for frame_idx in range(1,201):
            frame_idx = str(frame_idx)
            boxes_list = []
            scores_list = []
            labels_list  = []
            weights = [1] * len(datas)
            weights[0] = 3

            for data in datas:
                data_box = []
                score_box = []
                label_box = []
                if video_id in data and int(frame_idx) in data[video_id]:
                    for box in data[video_id][int(frame_idx)]:
                        data_box.append(box[:4])
                        score_box.append(box[4])
                        label_box.append(box[5])
                boxes_list.append(data_box)
                scores_list.append(score_box)
                labels_list.append(label_box)

            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            for i in range(len(boxes)):
                results.append([video_id, frame_idx, boxes[i][0] *w , boxes[i][1] * h, (boxes[i][2] - boxes[i][0]) * w
                , (boxes[i][3] - boxes[i][1]) * h, labels[i], scores[i]])

    return results

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Inference')
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--checkpoint_path', type=str, default='weights')
    args.add_argument('--config_path', type=str, default='configs')
    args.add_argument('--p', type=float, default=0.0001)
    args.add_argument('--test_path', type=str, default='/data/aicity2024_track5_test/videos')
    args = args.parse_args()

    p = args.p
    batch_size = args.batch_size
    test_path = args.test_path
    config_path = args.config_path
    checkpoint_files = args.checkpoint_path
    print("Start inference")
    process_video_results = detect_video(test_path, config_path, checkpoint_files, batch_size)

    print("Start Fuse")
    results = fuse(process_video_results, test_path)

    print("Start Minority")
    minority_score = minority(p, results)

    # Remove boxes with score less than minority_score
    new_results = []
    for result in results:
        if result[-1] >= minority_score:
            new_results.append(result)
    results = new_results   

    print("Start Virtural Expander")
    results = Virtural_Expander(results)
    
    with open("results.txt", "w") as f:
        f.write(results)