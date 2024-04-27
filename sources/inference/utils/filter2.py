from utils.utils import overlap_ratio
from utils.detection_object import Head, Human, Motor
class Filter:
    def __init__(self, motorlist, humanlist) -> None:
        self.motorlist = motorlist
        self.humanlist = humanlist
        self.allclass = []
    def remove_overlap(self):
            list_to_remove = []
            for motor in self.motorlist:
                for motor2 in self.motorlist:
                    if motor.motor_id != motor2.motor_id and overlap_ratio(motor.get_box_info(), motor2.get_box_info()) > 0.9:
                        id_to_remove = motor.motor_id if motor.conf < motor2.conf else motor2.motor_id
                        list_to_remove.append(id_to_remove)
            self.motorlist = [motor for motor in self.motorlist if motor.motor_id not in list_to_remove]

            remove_list = {}
            for human in self.humanlist:
                for human2 in self.humanlist:
                    if human.human_id != human2.human_id  and human.class_id == human2.class_id and overlap_ratio(human.get_box_info(), human2.get_box_info()) > 0.9:
                        if human.class_id not in remove_list:
                            remove_list[human.class_id] = []

                        id_to_remove = human.human_id if human.conf < human2.conf else human2.human_id
                        remove_list[human.class_id].append(id_to_remove)
            for key in remove_list:
                self.humanlist = [human for human in self.humanlist if human.human_id not in remove_list[key]]
    def create_virtual(self):
        self.remove_overlap()
        class_list = ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0','8.0','9.0']
        for motor in self.motorlist:
            self.allclass.append(motor)
            left, top, right, bottom,  class_id, conf, cls_conf = motor.get_box_info()
            for cl in class_list:
                if float(cl) != class_id:
                    self.allclass.append(Human(bbox=[left, top, right-left, bottom-top, float(cl), 0.00001]))
        for human in self.humanlist:
            self.allclass.append(human)
            left, top, right, bottom,  class_id, conf, cls_conf = human.get_box_info()
            for cl in class_list:
                if float(cl) != class_id:
                    if float(cl) == 1.0:
                        self.allclass.append(Human(bbox=[left, top, right-left, bottom-top, float(cl), 0.00001]))
                    else:
                        self.allclass.append(Human(bbox=[left, top, right-left, bottom-top, float(cl), 0.001]))
        return self.allclass

