
CLASSES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

CLASS_DICT = dict(zip(CLASSES, [i for i in range(1,6)]))
print(CLASS_DICT)
DATA_PATH = '/home/xie/PycharmProjects/Migration_learning/flower_photos'
IMAGE_SIZE = 224
FLIPPED = True
BATCH_SIZE = 45
EPOCH =20

