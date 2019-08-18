import migration_config as cfg
import os
import numpy as np

def rename_image(path):
    for file in os.listdir(path):
        if file.endswith('.txt'):
            continue
        #class_number = class_dict[file]
        #print(file)
        class_path = os.path.join(path, file)
        print(class_path)
        n = 1
        for image in os.listdir(class_path):
            print(n)
            old_name = os.path.join(class_path, image)
            new_name = os.path.join(class_path, str(n)+'.jpg')
            os.rename(old_name, new_name)
            n += 1

def create_index(path,class_dict):
    train = open(os.path.join(path, 'train.txt'), 'w')
    test = open(os.path.join(path, 'test.txt'), 'w')
    for file in os.listdir(path):
        class_path = os.path.join(path, file)
        if os.path.isfile(class_path):
            continue
        print(class_path)
        class_number = class_dict[file]
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path, image)
            chance = np.random.randint(100)
            if chance < 10:
                test.write(image_path+' '+str(class_number)+'\n')
            else:
                train.write(image_path+' '+str(class_number)+'\n')
    train.close()
    test.close()



if __name__ == '__main__':
    #rename_image(cfg.DATA_PATH, cfg.CLASS_DICT)
    create_index(cfg.DATA_PATH, cfg.CLASS_DICT)



