import os
import os.path as osp
import sys
import json
from PIL import Image
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import pdb
import numpy as np

def normalize():
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Normalize(mean, std)

class MLLM_Tester(nn.Module):
    def __init__(self):
        super().__init__()
        # initialization of qwen_vl_chat
        qwen_vl_chat_path = "Qwen-VL-Chat path"
        self.model = AutoModelForCausalLM.from_pretrained(
            qwen_vl_chat_path, device_map='cuda', trust_remote_code=True).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_vl_chat_path,
                                                trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(qwen_vl_chat_path, trust_remote_code=True)
        self.model.generation_config.top_p = 0.01

        self.vis_processor = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                normalize(),
            ]
        )

    def forward(
        self,
        x
    ):
        data_path, prompt, candidates = x['data_path'], x['question'], x['choices']
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
        image = image.unsqueeze(0)

        all_losses = []
        for cand in candidates:
            if "<img>" not in prompt:
                if type(data_path) is not list:
                    question_info_list = [
                        {'image':data_path},
                        {'text': prompt},
                    ]  
                    info_list = [
                        {'image':data_path},
                        {'text': prompt + cand},
                    ]
                else:
                    question_info_list = []
                    info_list = []
                    for current_path in data_path:
                        question_info_list.append({'image':current_path})
                        info_list.append({'image':current_path})
                    question_info_list.append({'text': prompt})
                    info_list.append({'text': prompt + cand})
            else:
                text_list = prompt.split("<img>")
                question_info_list = [{'text': text_list[0]}]
                info_list = [{'text': text_list[0]}]
                for index in range(len(data_path)):
                    question_info_list.append({'image':data_path[index - 1]})
                    info_list.append({'image':data_path[index - 1]})
                    question_info_list.append({'text': text_list[index]})
                    if index != len(data_path) - 1 :
                        info_list.append({'text': text_list[index]})
                    else:
                        info_list.append({'text': text_list[index] + cand})

            question_query = self.tokenizer.from_list_format(question_info_list)
            question_tokenized = self.tokenizer(question_query)
            query = self.tokenizer.from_list_format(info_list)
            tokenized = self.tokenizer(query)
            num_masked_tokens = len(question_tokenized['input_ids'])
            input_ids = torch.tensor(tokenized.input_ids).unsqueeze(0).cuda()
            attention_mask = torch.tensor(tokenized.attention_mask).unsqueeze(0).cuda()
            target_len = input_ids.shape[1] - num_masked_tokens
            labels = torch.clone(input_ids)
            labels[0,:num_masked_tokens] = -100

            with torch.no_grad():
                output = self.model(input_ids = input_ids[:,:-1], attention_mask = attention_mask[:,:-1], return_dict=True)
                loss = torch.nn.functional.cross_entropy(output.logits.permute(0, 2, 1), input_ids[:,1:].cuda(),reduction='none')
                all_losses.append(loss[0, -target_len:].mean().item())

        return all_losses


def build():
    return MLLM_Tester()