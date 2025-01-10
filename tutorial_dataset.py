import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        # with open('../SelfData/image_descriptions_cleaned.json', 'rt') as f:  #fill50k  SelfData image_descriptions_cleaned
        with open('double_input/first_40000_prompt.json','rt') as f:  # fill50k  SelfData image_descriptions_cleaned
            for line in f:
                self.data.append(json.loads(line))

            # file_content = f.read()
            # data = json.loads(file_content)
            # print("data:",data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        mask_filename="mask/"+item['target'].split('/')[1].split('.')[0]+'_masked_inverted.png'
        # print("mask_filename:",mask_filename)
        # print("source_filename:",source_filename)
        # print("target_filename:",target_filename)
        source = cv2.imread('double_input/' + source_filename)
        target = cv2.imread('double_input/' + target_filename)
        mask = cv2.imread('double_input/' + mask_filename)
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))
        mask=cv2.resize(mask, (512, 512))
        # print("source:",source.shape)
        # print("target:", target.shape)
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        source=source+mask
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

