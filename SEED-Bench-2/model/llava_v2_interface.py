import os
import os.path as osp
import sys
import json
from PIL import Image
import torch
import copy

sys.path.append("/LLaVA/path")

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import pdb
import numpy as np

import requests
from PIL import Image
from io import BytesIO
import re

def normalize():
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Normalize(mean, std)

class MLLM_Tester(nn.Module):
    def __init__(self):
        super().__init__()
        # initialization of llava 1.5
        model_path = "/llava_1.5/model_path"

        model_base = None
        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name)
        self.vis_processor = transforms.Compose(
            [
                transforms.Resize(
                    (336, 336), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                normalize(),
            ]
        )

    def forward(
        self,
        x
    ):
        data_path, question, choices = x['data_path'], x['question'], x['choices']
        if type(data_path) != list:
            with Image.open(data_path) as raw_image:
                raw_image = raw_image.convert("RGB")
                image = self.vis_processor(raw_image).unsqueeze(0).cuda()
        else:
            image = []
            for i in range(len(data_path)):
                with Image.open(data_path[i]) as raw_image:
                    raw_image = raw_image.convert("RGB")
                    image.append(self.vis_processor(raw_image).unsqueeze(0).cuda())
            image = torch.cat(image, dim=0)

        if "[img]" in question:
            question = question.replace("[img]", IMAGE_PLACEHOLDER)
        
        all_losses = []
        with torch.no_grad():
            for cand in choices:
                qs = copy.deepcopy(question)
                image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                if IMAGE_PLACEHOLDER in qs:
                    if self.model.config.mm_use_im_start_end:
                        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                    else:
                        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
                else:
                    if self.model.config.mm_use_im_start_end:
                        qs = image_token_se + "\n" + qs
                    else:
                        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                conv_mode = "llava_v1"

                from llava.conversation import conv_templates, SeparatorStyle
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], cand)

                answer_input_id = torch.tensor(self.tokenizer(cand).input_ids).unsqueeze(0).cuda()
                prompt = conv.get_prompt()

                input_ids = (
                    tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )
                num_mask = answer_input_id.shape[1]
                labels = input_ids.clone()
                labels[:,:-1 * (num_mask)] = -100
                loss = self.model(input_ids, images=image.to(torch.float16), labels=labels).loss
                # print("loss:", loss)
                all_losses.append(loss.item())
        
        return all_losses



def build():
    return MLLM_Tester()