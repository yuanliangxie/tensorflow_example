import cv2
import migration_config as cfg
import numpy as np
import os
import pickle
import random



class pascal_voc_xie(object):
    def __init__(self, phase=None, rebuild=False):
        self.image_size = cfg.IMAGE_SIZE
        self.obj_class = cfg.CLASSES
        self.flipped = cfg.FLIPPED
        self.batch_size = cfg.BATCH_SIZE
        self.data_path = cfg.DATA_PATH
        self.train_index = os.path.join(self.data_path, 'train.txt')
        self.test_index = os.path.join(self.data_path, 'test.txt')
        self.phase = phase
        self.epoch = 1
        self.num = 0
        self.index = None
        self.rebuild = rebuild
        self.is_rebuild()
        self.prepare_index()


    def what_your_need(self):
        if self.phase == 'train':
            index_file = self.train_index
            if self.flipped:
                pkl_file = os.path.join(self.data_path, 'data_train_flipped.pkl')
            else:
                pkl_file = os.path.join(self.data_path, 'data_train.pkl')
        elif self.phase == 'test':
            index_file = self.test_index
            if self.flipped:
                pkl_file = os.path.join(self.data_path, 'data_test_flipped.pkl')
            else:
                pkl_file = os.path.join(self.data_path, 'data_test.pkl')
        else:
            return None, None
        return index_file, pkl_file

    def is_rebuild(self):
        _, pkl_file = self.what_your_need()
        try:
            if os.path.isfile(pkl_file):
                if self.rebuild:
                    os.remove(pkl_file)
                else:
                    pass
            else:
                pass
        except TypeError:
            print('warning:phase is not None!')

    def prepare_index(self):
        index_file, pkl_file = self.what_your_need()
        if not index_file and not pkl_file:
            print('路径装填不成功！')
            return
        if os.path.isfile(pkl_file):
            print('Loading index from: '+pkl_file)
            with open(pkl_file, 'rb') as f:
                self.index = pickle.load(f)
                random.shuffle(self.index)
            return
        else:
            with open(index_file, 'r') as f:
                index = [index.strip('\n').split(' ') for index in f.readlines()]
                noflipped_mark = [0]*len(index)
                index_path, class_number = list(zip(*index))
                index_0 =list(zip(index_path, class_number, noflipped_mark))
                if self.flipped:
                    flipped_mark = [1]*len(index)
                    index_1 = list(zip(index_path, class_number, flipped_mark))
                    self.index = index_0+index_1
                else:
                    self.index = index_0
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.index, f)





    def get_index(self):
            if (self.num + self.batch_size) > len(self.index):
                self.epoch = self.epoch + 1
                self.num = 0
                random.shuffle(self.index)#是对list进行shuffle
            index = self.index[self.num: self.num + self.batch_size]
            self.num = self.num + self.batch_size
            return index

    def get_image(self, index, choose_flipped = False):
        image = np.zeros([self.batch_size, self.image_size, self.image_size, 3], dtype=np.uint8)
        for i,  index in enumerate(index):
            if index[2] == 0 and not choose_flipped:
                pic_name = index[0]
                pic = cv2.imread(pic_name)
                pic = cv2.resize(pic, (self.image_size, self.image_size))
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                image[i, :, :, :] = pic
                # cv2.imshow('flowre', image[i, :, :, :])
                # cv2.waitKey(20)#展示加载过程
            else:
                pic_name = index[0]
                pic = cv2.imread(pic_name)
                pic = cv2.resize(pic, (self.image_size, self.image_size))
                pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
                image[i, :, ::-1, :] = pic

        return image#/255.0*2.0 -1.0

    def get_label(self, index):
        label = np.zeros([self.batch_size, 1])
        for i, index in enumerate(index):
            label[i, :] = index[1]
        return label

    def get(self, chooose_flipped = False):#chooose_flipped = True时为只给出左右翻转后的图片
        index = self.get_index()
        return self.get_image(index, choose_flipped=chooose_flipped), self.get_label(index)

    def reset_num(self):
        self.num = 0

    def reset_epoch(self):
        self.epoch = 1




#存储字典
# for index in self.image_index:
#             label, num = self.load_pascal_annotation(index)
#             if num == 0:
#                 continue
#             imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
#             gt_labels.append({'imname': imname,
#                               'label': label,
#                               'flipped': False})
#         print('Saving gt_labels to: ' + cache_file)
#         with open(cache_file, 'wb') as f:
#             pickle.dump(gt_labels, f)
#提取字典
# with open(cache_file, 'rb') as f:
#     gt_labels = pickle.load(f)
if __name__ == '__main__':
    solver = pascal_voc_xie(phase='train')
    index = solver.index
    image, label =solver.get(chooose_flipped=True)#image.np的dtype为np.uint8
    #image = np.cast(image, np.uint8)
    print(label[0, :])
    cv2.imshow('image', image[0, :, :, :])
    cv2.waitKey()
    cv2.destroyWindow('image')