import time
import math

import cv2
import torch
import torchvision
import numpy as np


COLORS = [(255, 179, 0),
        (128, 62, 117),
        (255, 104, 0),
        (166, 189, 215),
        (193, 0, 32),
        (206, 162, 98),
        (129, 112, 102),
        (0, 125, 52),
        (246, 118, 142),
        (0, 83, 138),
        (255, 122, 92),
        (83, 55, 122),
        (255, 142, 0),
        (179, 40, 81),
        (244, 200, 0),
        (127, 24, 13),
        (147, 170, 0),
        (89, 51, 21),
        (241, 58, 19),
        (35, 44, 22)]


class_dict = [
    'motorbike',
    'DHelmet',
    'DNoHelmet',
    'P1Helmet',
    'P1NoHelmet',
    'P2Helmet',
    'P2NoHelmet',
]

def get_class_by_names(bboxes, labels, classes=class_dict):
    """Get all the bboxes information coresponding with all class in labels list

    Args:
        bboxes (np.array, list): list of all the objects bounding boxes information
        labels (list, str): list of labels to get their indexes
        classes (list, string): classes list

    Returns:
        labels_indexes: found indexes
    """
    
    bboxes = np.array(bboxes)
    labels_indexes = np.empty(0)
    for label in labels:
        label_indexes = np.where(bboxes[..., -1] == classes.index(label))
        if len(label_indexes):
            labels_indexes = np.concatenate((labels_indexes, label_indexes[0])).astype(int)
    return bboxes[labels_indexes]


def get_original_box(path, order='class_last'):
    original_bbox_info = {}
    f = open(path, "r")
    for line in f.readlines():
        data = line.split(',')
        video_id, frame, bb_left, bb_top, bb_width, bb_height, class_id, class_confidence = int(data[0]), int(data[1]), \
            int(data[2]), int(data[3]), int(data[4]), int(data[5]), int(data[6]), float(data[7].split('\n')[0])
        if video_id not in original_bbox_info.keys():
            original_bbox_info[video_id] = {}
        if frame not in original_bbox_info[video_id].keys():
            original_bbox_info[video_id][frame] = []
        if order == 'class_last':
            original_bbox_info[video_id][frame].append([bb_left, bb_top, bb_left + bb_width, bb_top + bb_height, class_confidence, class_id - 1])
        else:
            original_bbox_info[video_id][frame].append([bb_left, bb_top, bb_left + bb_width, bb_top + bb_height, class_id - 1, class_confidence])
    return original_bbox_info

def draw_image(image, results, use_track=False):
    """Draw bbox and text on the image
    Args:
        image (cvMat): Image opencv mat
        label (list - string): [video_id, frame, track_id, bb_left, bb_top, bb_right, bb_bottom, class_id]
    """
    h, w, c = image.shape
    if use_track:
        for left, top, right, bottom, class_id, confidence, track_id, direction, motion, _ in results:
            if left <=1 and top <= 1 and right <= 1 and bottom <= 1:
                left, top, right, bottom = left * w, top * h, right * w, bottom * h
            left, top, right, bottom, class_id = int(left), int(top), int(right), int(bottom), int(class_id)
            cv2.rectangle(image, (left, top), (right, bottom), colors[class_id])
            if motion == 1:
                motion_text = '-' + 'Motion'  
            elif motion == 0:    
                motion_text = '-' + 'Stop'  
            else :
                motion_text = ''
            text = class_dict[class_id] + '-' + str(round(confidence, 2)) + motion_text + '_' + str(track_id)
            cv2.putText(image, text, (left, top - 10), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_id], 1, cv2.LINE_AA)
                
    else:
        for left, top, right, bottom, confidence, class_id in results:
            if left <=1 and top <= 1 and right <= 1 and bottom <= 1:
                left, top, right, bottom = left * w, top * h, right * w, bottom * h
            left, top, right, bottom, class_id = int(left), int(top), int(right), int(bottom), int(class_id)
            cv2.rectangle(image, (left, top), (right, bottom), colors[class_id])
            cv2.putText(image, class_dict[class_id] + '-' + str(round(confidence, 2)), (left, top - 10), \
                cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_id], 1, cv2.LINE_AA)
                

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    box1, box2 = torch.tensor(box1), torch.tensor(box2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y



def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    
    prediction = torch.tensor(prediction)

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = x[:, :4].clone()  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes
        import pdb; pdb.set_trace()
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def overlap_ratio(boxA, boxB):
    """Calculate iou of 2 bboxes

    Args:
        boxA (list): [x1, y1, x2, y2]
        boxB (list): [x1, y1, x2, y2]

    Returns:
        overlap_ratio (float): overlap ratio max(interArea / float(boxBArea), interArea / float(boxAArea))
    """
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if boxBArea == 0 or boxAArea == 0:
        return 0
    overlap_ratio = max(interArea / float(boxBArea), interArea / float(boxAArea))
    # return the overlap ratio
    return overlap_ratio     




def remove_occluded_objects(objects, classes, remove_threshold=0.6):
    """Remove occluded objects

    Args:
        objects (np.array): list of objects objects
        remove_threshole (float): if overlap_areas > remove_threshold then remove the motor
    """
    
    objects = get_class_by_names(objects, labels=classes)
    remove_indexes = []
    num_objects = len(objects)
    # import pdb; pdb.set_trace()
    for i in range(0, num_objects - 1):
        for j in range(i+1, num_objects):
            if overlap_ratio(objects[i],objects[j]) > remove_threshold:
                if objects[i][-2] > objects[j][-2]:
                    remove_indexes.append(j)
                else:
                    remove_indexes.append(i)
    remove_indexes = list(set(remove_indexes))
    objects = np.delete(objects, remove_indexes, axis=0)
    return objects

def remove_small_object(bboxes, image_w, image_h, min_size=40):
    widths = (bboxes[:,2] - bboxes[:,0]) * image_w
    heights = (bboxes[:,3] - bboxes[:,1]) * image_h
    keep_indexes = (widths > 40) * (heights > 40)
    return bboxes[keep_indexes]


def remove_ignore_objects(bboxes1, bboxes2, threshold=0.2):
    """Find and remove ignore objects bboxes1 and bboxes2

    Args:
        bboxes1 (list): [ bb_left, bb_top, bb_right, bb_bottom, cls_conf, cls_id ]
        bboxes2 (list): [ bb_left, bb_top, bb_right, bb_bottom, cls_conf, cls_id ]
    Return:

    """                
    if len(bboxes1) and len(bboxes2):
        ious = box_iou(bboxes1[:,:-2], bboxes2[:,:-2])
        keep_indexes = np.argwhere(ious > threshold)
        bboxes1, bboxes2 = bboxes1[keep_indexes[:, 0]], bboxes2[keep_indexes[:, 1]]
    return np.concatenate((bboxes1, bboxes2), axis=0)


def remove_redundant_objects(objects, remove_threshold=0.8):
    """Remove redundant motor objects

    Args:
        objects (np.array): list of objects objects
        remove_threshole (float): if overlap_areas > remove_threshold then remove the motor
    """
    remove_indexes = []
    num_motor = len(objects)
    # import pdb; pdb.set_trace()
    for i in range(0, num_motor - 1):
        for j in range(i+1, num_motor):
            if overlap_ratio(objects[i].get_box_info(),objects[j].get_box_info()) > remove_threshold:
                boxAArea_conf = objects[i].get_box_info()[-2]
                boxBArea_conf = objects[j].get_box_info()[-2]
                if boxAArea_conf < boxBArea_conf:
                    remove_indexes.append(i)
                else:
                    remove_indexes.append(j)
    remove_indexes = list(set(remove_indexes))
    keep_objects = []
    for i in range(len(objects)):
        if i not in remove_indexes:
            keep_objects.append(objects[i])
    return keep_objects

class VideoWriter(object):
    def __init__(self, save_path):
        frame_width = 1920
        frame_height = 1080
        size = (frame_width, frame_height)
        self.video_writer = cv2.VideoWriter(save_path, 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
        
    def write(self, frame):
        self.video_writer.write(frame)
        
    def destroy(self):
        self.video_writer.release()