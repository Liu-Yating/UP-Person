import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset
import re

def pre_caption(caption):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')
        
    return caption

class FLICKR30K(BaseDataset):
    """
    FLICKR30K
    """
    dataset_dir = 'flickr30k'

    def __init__(self, root='', verbose=True):
        super(FLICKR30K, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = self.dataset_dir

        self.anno_path_train = op.join(self.dataset_dir, 'flickr30k_train.json')
        self.anno_path_test = op.join(self.dataset_dir, 'flickr30k_test.json')
        self.anno_path_val = op.join(self.dataset_dir, 'flickr30k_val.json')
        self._check_before_run()

        self.train_annos = self._split_anno_train(self.anno_path_train)
        self.test_annos = self._split_anno_test(self.anno_path_test)
        self.val_annos = self._split_anno_val(self.anno_path_val)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> FLICKR30K Images and Captions are loaded")
            self.show_dataset_info()

    def _split_anno_train(self, anno_path: str):
        train_annos = []
        annos = read_json(anno_path)
        for anno in annos:
            train_annos.append(anno)
        return train_annos
    
    def _split_anno_test(self, anno_path: str):
        test_annos = []
        annos = read_json(anno_path)
        for anno in annos:
            test_annos.append(anno)
        return test_annos

    def _split_anno_val(self, anno_path: str):
        val_annos = []
        annos = read_json(anno_path)
        for anno in annos:
            val_annos.append(anno)
        return val_annos


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos



    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            for anno in annos:
                image_id = int(anno['image_id'])  
                pid = image_id
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['image'])
                caption = anno['caption'] 
                dataset.append((pid, image_id, img_path, pre_caption(caption)))

            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            image_id = 0
            for anno in annos:
                pid = image_id
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['image'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['caption'] # caption list
                for caption in caption_list:
                    captions.append(pre_caption(caption))
                    caption_pids.append(pid)
                image_id += 1
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path_train):
            raise RuntimeError("'{}' is not available".format(self.anno_path_train))
