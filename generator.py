
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from   PIL import Image
from   functools import partial


class UdacityDataset():

    @staticmethod
    def grayscale(rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    @staticmethod
    def saturation(rgb, saturation_var=0.5):
        gs = UdacityDataset.grayscale(rgb)
        alpha = 2 * np.random.random() * saturation_var 
        alpha += 1 - saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    @staticmethod
    def brightness(rgb, saturation_var=0.5):
        alpha = 2 * np.random.random() * brightness_var 
        alpha += 1 - saturation_var
        rgb   = rgb * alpha
        return np.clip(rgb, 0, 255)

    @staticmethod
    def contrast(rgb, contrast_var=0.5):
        gs = UdacityDataset.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * contrast_var 
        alpha += 1 - contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    @staticmethod
    def lighting(img, lighting_std=0.5):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)

    @staticmethod
    def none(img):
        return img

    
    def cls2onehot(self, idx):
        res      = np.zeros((len(self.labels)-1,))
        res[idx] = 1
        return res

    def __init__(self, 
                 data_dir, 
                 config, 
                 new_size, 
                 num_examples=-1, 
                 augmentation=[UdacityDataset.none], 
                 priors_pkl='prior_boxes_ssd300.pkl'):
        super(UdacityDataset, self).__init__()
        self.data_dir      = data_dir
        self.config        = config
        self.default_boxes = default_boxes
        self.new_size      = new_size
        self.augmentation  = list(set(augmentation)|set([UdacityDataset.none]))
        self.labels        = ['car', 'truck', 'pedestrian', 'bicyclist',  'traffic light']
        self.priors        = pickle.load(open(priors_pkl, 'rb'))
        self.bbox_util     = BBoxUtility(len(self.labels)+1, self.priors)
        self.ids           = {
            'all': [
                x 
                for x in map(lambda x: x, os.listdir(self.data_dir)) 
                if 'jpg' in x
            ][:num_examples]
        }
        self.ids['train'] = self.ids['all'][:int(len(self.ids['all']) * 0.75)]
        self.ids['val']   = self.ids['all'][int(len(self.ids['all'])  * 0.75):]

    def __len__(self):
        return len(self.ids['all'])

    def _get_image(self, index):
        return Image.open(
            os.path.join(self.data_dir, self.ids['all'][index])
        )
        return img

    def _get_annotation(self, index, hw):
        (h, w)   = hw
        boxes    = []
        labels   = []
        df       = pd.read_csv(
             os.path.join(self.data_dir, self.config)
        )
        for idx, row in df[df['frame']==self.ids['all'][index]].iterrows():
            xmin = float(row['xmin']) / w
            ymin = float(row['ymin']) / h
            xmax = float(row['xmax']) / w
            ymax = float(row['ymax']) / h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.cls2onehot(int(row['class_id'])))
        return np.hstack((boxes, labels))

    def generate(self, subset='all'):
        for index in range(len(self.ids[subset])):
            filename        = self.ids['all'][index]
            augmentation    = np.random.choice(self.augmentation)
            img             = self._get_image(index)
            annot_boxes     = self._get_annotation(index, img.size)
            img = np.array(
                img.resize(
                    (self.new_size, self.new_size), 
                    interpolation=cv2.INNER_NEAREST
                ), 
                dtype=np.float32
            )
            img              = augmentation(img)
            img              = (img / 127.0) - 1.0
            img              = tf.constant(img, dtype=tf.float32)
            annot_boxes      = self.bbox_util.assign_boxes(annot_boxes)
            yield filename, img,  annot_boxes

class UdacityGenerator:
    def __new__(cls, 
                data_dir='./udacity_driving_datasets', 
                config='labels_trainval.csv', 
                new_size=300, 
                batch_size=32, 
                num_batches=120,
                mode='train',
                augmentation=[UdacityDataset.none]):
    num_examples = batch_size * num_batches if num_batches > 0 else -1
    udacity      = UdacityDataset(
        data_dir, config, new_size, num_examples, augmentation
    )
    info = {
        'labels':    udacity.labels,
        'n_classes': len(udacity.labels),
        'shape':     (new_size, new_size, 3),
        'length':    len(udacity)
    }
    if mode == 'train':
        train_gen     = partial(udacity.generate, subset='train')
        train_dataset = tf.data.Dataset.from_generator(
            train_gen, (tf.string, tf.float32, tf.int64, tf.float32)
        )
        val_gen       = partial(udacity.generate, subset='val')
        val_dataset   = tf.data.Dataset.from_generator(
            val_gen, (tf.string, tf.float32, tf.int64, tf.float32)
        )
        train_dataset = train_dataset.shuffle(40).batch(batch_size)
        val_dataset   = val_dataset.batch(batch_size)
        return train_dataset.take(num_batches), val_dataset.take(-1), info
    else:
        dataset       = tf.data.Dataset.from_generator(
            udacity.generate, (tf.string, tf.float32, tf.int64, tf.float32)
        )
        dataset       = dataset.batch(batch_size)
        return dataset.take(num_batches), info