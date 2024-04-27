import uuid
from utils import overlap_ratio

import numpy as np

class Motor:
    
    def __init__(self, bbox=None, cls_conf=-1, combine_expand=0.05) -> None:
        """Initial for human object

        Args:
            bbox (np.array | list): Defaults to None.
            cls_conf (float) : None if object is not used for classification
        """
        self.left, self.top, self.width, self.height, self.class_id, self.conf \
            = np.array(bbox).astype(float)
        self.cls_conf = cls_conf
        self.motor_id = str(uuid.uuid4().int)
        self.human_id = None
        self.humans = []
        self.heads = []
        self.right  = self.width + self.left
        self.bottom = self.height + self.top
        self.combine_expand_w = combine_expand * self.width
        self.combine_expand_h = combine_expand * self.height
        self.combined_box = [self.left, self.top, self.right, self.bottom]
        self.type = "motor"
    
    def get_box_info(self):
        """Return object info

        Returns:
            list: [motor_id, left, top, right, bottom, class_id, conf, cls_conf]
        """
        return [self.left, self.top, self.right, self.bottom,  self.class_id, self.conf, self.cls_conf]
        

class Human(Motor):
    
    def __init__(self, bbox=None, cls_conf=-1, overlap_thres=0.3) -> None:
        """Initial for human object

        Args:
            bbox (np.array | list): . Defaults to None.
            cls_conf (float) : None if object is not used for classification
            overlap_thres (float): thres to decide human is belong to motor
        """
        super().__init__(bbox=bbox, cls_conf=cls_conf)
        self.human_id = str(uuid.uuid4().int)
        self.motor_id = None
        self.overlap_thres = overlap_thres
        self.wear_helmet = False
        self.x_center = (self.left + self.right) / 2
        self.heads = []
        self.type = "human"
        
    def attach_motor_id(self, motors: list):
        """Attach motor id to human object

        Args:
            motors (list): list of motor object
        """
        for motor in motors:
            overlap = overlap_ratio(self.get_box_info(),motor.get_box_info())
            if overlap > self.overlap_thres:
                self.motor_id = motor.motor_id
                motor.humans.append(self)
                motor.combined_box = [max(0, min(self.left, motor.combined_box[0]) - self.combine_expand_w), 
                                        max(0, min(self.top, motor.combined_box[1]) - self.combine_expand_h), 
                                        min(1920 ,max(self.right, motor.combined_box[2]) + self.combine_expand_w), 
                                        min(1080, max(self.bottom, motor.combined_box[3]) + self.combine_expand_h)]
                break
    
    def attach_head_id(self, heads: list):
        """Find nearest head to the human and attach head id to human

        Args:
            heads (list): list of Head objects
        """
        keep_heads = []
        for head in heads:
            overlap = overlap_ratio(self.get_box_info(),head.get_box_info())
            if overlap > self.overlap_thres:
                keep_heads.append(head)
        
        if len(keep_heads):
            min_dis = self.right - self.left
            nearest_head_index = -1
            for i, head in enumerate(keep_heads):
                human_head_dis = abs(self.x_center - head.x_center)
                if human_head_dis < min_dis:
                    min_dis = human_head_dis
                    nearest_head_index = i
                    
            keep_heads[nearest_head_index].motor_id = self.motor_id
            keep_heads[nearest_head_index].human_id = self.human_id
            self.wear_helmet = keep_heads[nearest_head_index].is_helmet
            self.cls_conf = keep_heads[nearest_head_index].cls_conf
            self.heads.append(keep_heads[nearest_head_index])
            
            
class Head(Motor):
    
    def __init__(self, bbox=None, cls_conf=-1, is_helmet=False, overlap_thres=0.6) -> None:
        """Initial for human object

        Args:
            bbox (np.array | list): . Defaults to None.
        """
        super().__init__(bbox=bbox, cls_conf=cls_conf)
        self.motor_id = None
        self.human_id = None
        self.overlap_thres = overlap_thres
        self.is_helmet = True if self.class_id == 1 else False
        self.x_center = (self.left + self.right) / 2
        self.type = "head"
    
    def attach_motor_id(self, motors: list,head_motor_overlap_thresh):
        """Find nearest head to the human and attach head id to human

        Args:
            heads (list): list of Head objects
        """
        avg_overlaps = []
        for motor in motors:
            head_motor_overlap = overlap_ratio(self.get_box_info(),motor.get_box_info())
            if head_motor_overlap > head_motor_overlap_thresh:
                avg_overlaps.append(0)
                continue
            overlap = overlap_ratio(self.get_box_info(),motor.combined_box)
            if overlap > self.overlap_thres:
                for human in motor.humans:
                    overlap += overlap_ratio(self.get_box_info(),human.get_box_info())
                if len(motor.humans):
                    avg_overlaps.append(overlap/(len(motor.humans)))
                else:
                    avg_overlaps.append(0)
            else:
                avg_overlaps.append(0)
        if sum(avg_overlaps) > 0:
            max_index = avg_overlaps.index(max(avg_overlaps))
            self.motor_id = motors[max_index].motor_id 
            motors[max_index].heads.append(self)
            
