# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/v7w/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    # 'train': 'train2014',
    # 'valid': 'val2014',
    # 'test': 'test2015',
    'train': 'train',
    'valid': 'valid',
    'test': 'test',
}


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }


    A Visual7W data example in json file:
    {
        'image_id': 2359297,
        'question': 'Where is he sitting?',
        'multiple_choices': ['At a park.', 'On the grass.', 'At a dining table.'],
        'qa_id': 467550,
        'answer': 'On a bench.',
        'type': 'where'
    }
    """
    def __init__(self, split: str):
        # split: either "train", "val", or "test"
        self.name = split
        self.split = split

        # Loading datasets
        self.data = []
        self.data.extend(json.load(open("data/v7w/%s.json" % split)))
        print("Load %d data from split %s." % (len(self.data), self.name))


    @property
    def num_answers(self):
        return 4

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset


        # Loading detection features to img_data
        img_data = []


        if args.tiny:
            topk = TINY_IMG_NUM
        else:
            topk = None

        # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
        # It is saved as the top 5K features in val2014_***.tsv
        # load_topk = 5000 if (split == 'minival' and topk is None) else topk
        load_topk = topk
        img_data.extend(load_obj_tsv(
            os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[dataset.split])),
            topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            # example of img_datum['img_id'] = 'v7w_2375420'
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        drop_count = 0
        for datum in self.raw_dataset.data:
            coco_img_id = 'v7w_' + str(datum['image_id'])
            if coco_img_id in self.imgid2img:
                self.data.append(datum)
            else:
                drop_count += 1
        print("Only kept the data with loaded image features")
        print("Use %d data in torch dataset, %d dropped" % (len(self.data), drop_count))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = 'v7w_' + str(datum['image_id'])
        ques = datum['question']
        ques_type = datum['type']
        # ques = ques.replace(ques_type.title(), '')
        qa_id = datum['qa_id']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Merge question and multiple choices
        answer = datum['answer']
        candidates = datum['multiple_choices']
        # Example: answer = "abc", multiple_choices = ['ttt', 'vvv', 'zzz']
        # We want all choices to be "<0ttt <1vvv <2abc <3zzz", 
        # where the index of answer is randomly chosen between 0 and 3
        answer_idx = np.random.randint(0, 4)
        all_choices_raw = candidates[:answer_idx] + [answer] + candidates[answer_idx:]
        # Add indices to all_choices_raw
        all_choices = []
        for choice_idx, choice_raw in enumerate(all_choices_raw):
            temp = "<" + str(choice_idx) + choice_raw
            all_choices.append(temp)
        # Example of ques: 'Where is he sitting? <0At a park. <1On the grass. <2At a dining table. <3On a bench.'
        ques = ques + " " + " ".join(all_choices)
        target = answer_idx
        return feats, boxes, ques, target, ques_type, qa_id



class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset

    def evaluate(self, preds_all, labels_all, ques_types=None):
        assert len(preds_all) == len(labels_all)
        score = 0.
        score_qtype, total_qtype = {}, {}
        if ques_types:
            for p, l, q_type in zip(preds_all, labels_all, ques_types):
                if q_type not in score_qtype:
                    score_qtype[q_type] = 0
                    total_qtype[q_type] = 0
                total_qtype[q_type] += 1
                if p == l:
                    score += 1
                    score_qtype[q_type] += 1
            for key, val in score_qtype.items():
                score_qtype[key] = val / total_qtype[key]
            print(score_qtype)
            print(total_qtype)
        else:
            for p, l in zip(preds_all, labels_all):
                if p == l:
                    score += 1
        return score / len(preds_all)

    def dump_result(self, sentences, answers, labels, qa_ids, path):
        """
        Dump results to a json file.

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for sent, ans, label, qa_id in zip(sentences, answers, labels, qa_ids):
                result.append({
                    'qa_id': qa_id,
                    'sentence': sent,
                    'answer': ans,
                    'label': label
                })
            json.dump(result, f, indent=4, sort_keys=True)


