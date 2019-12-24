


import os
import random
import cv2
import numpy as np
from tensorpack.dataflow import DataFromGenerator,BatchData,MultiProcessPrefetchData
from lib.helper.logger import logger


from lib.dataset.augmentor.augmentation import ColorDistort,\
    Random_scale_withbbox,\
    Random_flip,\
    Fill_img,\
    Gray_aug,\
    baidu_aug,\
    dsfd_aug,\
    Pixel_jitter
from lib.core.model.facebox.training_target_creation import get_training_targets
from train_config import config as cfg


class data_info(object):
    def __init__(self,img_root,txt):
        self.txt_file=txt
        self.root_path = img_root
        self.metas=[]


        self.read_txt()

    def read_txt(self):
        with open(self.txt_file) as _f:
            txt_lines=_f.readlines()
        txt_lines.sort()
        for line in txt_lines:
            line=line.rstrip()

            _img_path=line.rsplit('| ',1)[0]
            _label=line.rsplit('| ',1)[-1]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label
            self.metas.append([current_img_path,current_img_label])

            ###some change can be made here
        logger.info('the dataset contains %d images'%(len(txt_lines)))
        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas

class FaceBoxesDataIter():
    def __init__(self,img_root_path='',ann_file=None,training_flag=True, shuffle=True):

        self.color_augmentor = ColorDistort()

        self.training_flag=training_flag

        self.lst=self.parse_file(img_root_path,ann_file)

        self.shuffle=shuffle



    def __call__(self, *args, **kwargs):
        idxs = np.arange(len(self.lst))

        if self.shuffle:
            np.random.shuffle(idxs)
        for k in idxs:
            yield self._map_func(self.lst[k], self.training_flag)



    def parse_file(self,im_root_path,ann_file):
        '''
        :return:
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()

        return all_samples


    def _map_func(self,dp,is_training):
        fname, annos = dp
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = annos.split(' ')
        boxes = []
        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            boxes.append(bbox)

        boxes = np.array(boxes, dtype=np.float)

        if is_training:
            # if random.uniform(0, 1)>0.7:
            image, boxes = Random_scale_withbbox(image, boxes, target_shape=[cfg.MODEL.hin, cfg.MODEL.win],
                                                     jitter=0.3)
            if random.uniform(0, 1) > 0.5:
                image, boxes = Random_flip(image, boxes)
            if random.uniform(0, 1) > 0.5:
                image=self.color_augmentor(image)

            if random.uniform(0, 1) > 0.5:
                image = Pixel_jitter(image, 15)
            if random.uniform(0, 1) > 0.8:
                image = Gray_aug(image)

        image, shift_x, shift_y = Fill_img(image, target_width=cfg.MODEL.win, target_height=cfg.MODEL.hin)
        # boxes[:, 0:4] = boxes[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
        boxes[:, 0] += shift_x
        boxes[:, 1] += shift_y
        boxes[:, 2] += shift_x
        boxes[:, 3] += shift_y
        boxes[boxes[:, 4]>=0, 4] += shift_x
        boxes[boxes[:, 5]>=0, 5] += shift_y
        boxes[boxes[:, 6]>=0, 6] += shift_x
        boxes[boxes[:, 7]>=0, 7] += shift_y
        boxes[boxes[:, 8]>=0, 8] += shift_x
        boxes[boxes[:, 9]>=0, 9] += shift_y
        boxes[boxes[:, 10]>=0, 10] += shift_x
        boxes[boxes[:, 11]>=0, 11] += shift_y
        boxes[boxes[:, 12]>=0, 12] += shift_x
        boxes[boxes[:, 13]>=0, 13] += shift_y

        h, w, _ = image.shape
        boxes[:, 0] /= w
        boxes[:, 1] /= h
        boxes[:, 2] /= w
        boxes[:, 3] /= h
        boxes[boxes[:, 4]>=0, 4] /= w
        boxes[boxes[:, 5]>=0, 5] /= h
        boxes[boxes[:, 6]>=0, 6] /= w
        boxes[boxes[:, 7]>=0, 7] /= h
        boxes[boxes[:, 8]>=0, 8] /= w
        boxes[boxes[:, 9]>=0, 9] /= h
        boxes[boxes[:, 10]>=0, 10] /= w
        boxes[boxes[:, 11]>=0, 11] /= h
        boxes[boxes[:, 12]>=0, 12] /= w
        boxes[boxes[:, 13]>=0, 13] /= h

        image = cv2.resize(image, (cfg.MODEL.win, cfg.MODEL.hin))
        image = image.astype(np.uint8)

        ### cover the small faces with invisible landmarks
        boxes_clean = []
        for i in range(boxes.shape[0]):
            box = boxes[i]
            if box[4] < 0:
                image[int(box[1]* cfg.MODEL.hin):int(box[3]* cfg.MODEL.hin), int(box[0]* cfg.MODEL.hin):int(box[2]* cfg.MODEL.hin), :] = cfg.DATA.PIXEL_MEAN
            else:
                boxes_clean.append([box[1], box[0], 
                                    box[3], box[2], 
                                    box[5], box[4], 
                                    box[7], box[6], 
                                    box[9], box[8], 
                                    box[11], box[10], 
                                    box[13], box[12]])
                

        boxes = np.array(boxes_clean)
        # tmp = boxes[:, ::2]
        # boxes[:, ::2] = boxes[:, 1::2]
        # boxes[:, 1::2] = tmp

        if cfg.TRAIN.vis:
            for i in range(boxes.shape[0]):
                box=boxes[i]
                color = (255, 0, 0)
                thickness = 2
                radius = 2
                cv2.rectangle(image, (int(box[1]*cfg.MODEL.win), int(box[0]*cfg.MODEL.hin)),
                                            (int(box[3]*cfg.MODEL.win), int(box[2]*cfg.MODEL.hin)), color, thickness)
                for point_x, point_y in zip(box[5::2], box[4::2]):
                    if point_x > 0 and point_y > 0:
                        cv2.circle(image, (int(point_x*cfg.MODEL.win), int(point_y*cfg.MODEL.hin)), radius, color, thickness)

        reg_targets, matches = self.produce_target(boxes)
        image = image.astype(np.float32)

        # if reg_targets.shape[0] > 0:
        #     reg_targets = reg_targets[:, :4]

        return image, reg_targets, matches

    def produce_target(self,bboxes):
        reg_targets, matches = get_training_targets(bboxes, threshold=cfg.MODEL.MATCHING_THRESHOLD)
        return reg_targets, matches

    def __len__(self):
        return len(self.lst)

class DataIter():
    def __init__(self,img_root_path='',ann_file=None,training_flag=True):
        self.training_flag=training_flag

        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size

        self.generator=FaceBoxesDataIter(img_root_path,ann_file,self.training_flag,)

        self.ds=self.build_iter()

        self.size=self.__len__()

    def build_iter(self):
        ds = DataFromGenerator(self.generator)

        ds = BatchData(ds, self.num_gpu *  self.batch_size)

        ds = MultiProcessPrefetchData(ds, self.prefetch_size, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds

    def __call__(self, *args, **kwargs):
        for i in range(self.size):
            tmp=next(self.ds)
            yield tmp[0],tmp[1],tmp[2]

    def __len__(self):
        return len(self.generator)//self.batch_size

    def _map_func(self,dp,is_training):
        raise NotImplementedError("you need implemented the map func for your data")

    def set_params(self):
        raise NotImplementedError("you need implemented  func for your data")