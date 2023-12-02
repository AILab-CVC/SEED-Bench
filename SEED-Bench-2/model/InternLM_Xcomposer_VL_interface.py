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
        # initialization of InternLM_Xcomposer_VL
        InternLM_Xcomposer_path = "./InternLM_Xcomposer_VL_weight_path"
        self.model = AutoModelForCausalLM.from_pretrained(
            InternLM_Xcomposer_path, device_map='cuda', trust_remote_code=True).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(InternLM_Xcomposer_path,
                                                trust_remote_code=True)
        for n, p in self.model.named_parameters():
            p.requires_grad = False

        self.model.tokenizer = self.tokenizer
        self.vis_processor = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                normalize(),
            ]
        )

    def process_qa(self, question, candidates):
        options = candidates
        options_prompt = f'A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n'  # noqa

        img_prompt = ' <|User|>:<ImageHere>'
        context = 'N/A'
        
        mid_prompt = 'Context: ' + context + '\nQuestion: ' + question + '\nOptions: []\n' + options_prompt
        ans_prompt = ' <|Bot|>: Answer: The answer is'
        text = img_prompt + mid_prompt + '<TOKENS_UNUSED_0>' + ans_prompt

        return text

    def process_interleved_qa(self, question, candidates):
        options = candidates
        options_prompt = f'A. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n'  # noqa

        img_prompt = ' <|User|>:'
        context = 'N/A'
        question = question.replace("[img]", '<ImageHere>')
        
        mid_prompt = 'Context: ' + context + '\nQuestion: ' + question + '\nOptions: []\n' + options_prompt
        ans_prompt = ' <|Bot|>: Answer: The answer is'
        text = img_prompt + mid_prompt + '<TOKENS_UNUSED_0>' + ans_prompt

        return text

    def generate_answer_with_ppl(self, base_prompt, image):
        choice_mapping = ['A.', 'B.', 'C.', 'D.']
        if type(image) is list:
            image_embeds = []
            for current_image in image:
                current_image_embed = self.model.encode_img(current_image)
                image_embeds.append(current_image_embed)
            img_embeds = torch.stack(image_embeds, dim=1).squeeze(0)
            N, L, C = img_embeds.shape
            img_embeds = img_embeds.reshape(1, -1, C)
        else:
            img_embeds = self.model.encode_img(image)
        prompt = base_prompt
        prompt_segs = prompt.split('<ImageHere>')
        prompt_seg_tokens = [
            self.model.tokenizer(seg,
                                return_tensors='pt',
                                add_special_tokens=i == 0).
            to(self.model.internlm_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        prompt_seg_embs = [
            self.model.internlm_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)

        im_targets = torch.ones(img_embeds.shape[0], img_embeds.shape[1], dtype=torch.long).to(img_embeds.device) * self.model.tokenizer.pad_token_id
        tars = torch.cat([prompt_seg_tokens[0], im_targets, prompt_seg_tokens[1]], dim=1)
        atts_mask = torch.ones(prompt_embs.size()[:-1], dtype=torch.long).to(self.model.device)

        len_prompt = tars.shape[1] - 1
        candis = choice_mapping
        op_tokens = self.model.tokenizer(candis, return_tensors="pt", add_special_tokens=False, padding=True).to(self.model.device)
        op_embeds = self.model.internlm_model.model.embed_tokens(op_tokens.input_ids)
        tars = torch.cat([tars.repeat(4,1), op_tokens.input_ids], dim=1)
        atts_mask = torch.cat([atts_mask.repeat(4,1), op_tokens.attention_mask], dim=1)
        prompt_embs = torch.cat([prompt_embs.repeat(4,1,1), op_embeds], dim=1)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = self.model.internlm_model(
                    inputs_embeds=prompt_embs,
                    attention_mask=atts_mask,
                    return_dict=True,
                    labels=None,
                )
        outputs = outputs.logits
        shift_logits = outputs[..., len_prompt:-1, :].contiguous()
        shift_labels = tars[..., 1+len_prompt:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.model.tokenizer.pad_token_id)
        sf = torch.nn.functional.softmax(shift_logits, dim=-1)

        loss = loss_fct(shift_logits, shift_labels.view(-1)).reshape(shift_labels.shape[0], shift_labels.shape[1])
        mask_length = None

        lens = (shift_labels !=
                self.model.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens

        return ce_loss

    def generate_interleved_answer_with_ppl(self, base_prompt, image):
        choice_mapping = ['A.', 'B.', 'C.', 'D.']
        if type(image) is list:
            image_embeds = []
            for current_image in image:
                current_image_embed = self.model.encode_img(current_image)
                image_embeds.append(current_image_embed)
            img_embeds = torch.stack(image_embeds, dim=1).squeeze(0)
            N, L, C = img_embeds.shape
            # img_embeds = img_embeds.reshape(1, -1, C)
        else:
            img_embeds = self.model.encode_img(image)
        prompt = base_prompt
        prompt_segs = prompt.split('<ImageHere>')
        prompt_seg_tokens = [
            self.model.tokenizer(seg,
                                return_tensors='pt',
                                add_special_tokens=i == 0).
            to(self.model.internlm_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        prompt_seg_embs = [
            self.model.internlm_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        im_targets = torch.ones(img_embeds.shape[0], img_embeds.shape[1], dtype=torch.long).to(img_embeds.device) * self.model.tokenizer.pad_token_id
        
        embs = [prompt_seg_embs[0]]
        tars = [prompt_seg_tokens[0]]
        for index in range(img_embeds.shape[0]):
            embs.append(img_embeds[index].unsqueeze(0))
            embs.append(prompt_seg_embs[index + 1])
            tars.append(im_targets[index].unsqueeze(0))
            tars.append(prompt_seg_tokens[index + 1])

        # prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(embs, dim=1)
        tars = torch.cat(tars, dim=1)
        # tars = torch.cat([prompt_seg_tokens[0], im_targets, prompt_seg_tokens[1]], dim=1)
        atts_mask = torch.ones(prompt_embs.size()[:-1], dtype=torch.long).to(self.model.device)

        len_prompt = tars.shape[1] - 1
        candis = choice_mapping
        op_tokens = self.model.tokenizer(candis, return_tensors="pt", add_special_tokens=False, padding=True).to(self.model.device)
        op_embeds = self.model.internlm_model.model.embed_tokens(op_tokens.input_ids)
        tars = torch.cat([tars.repeat(4,1), op_tokens.input_ids], dim=1)
        atts_mask = torch.cat([atts_mask.repeat(4,1), op_tokens.attention_mask], dim=1)
        prompt_embs = torch.cat([prompt_embs.repeat(4,1,1), op_embeds], dim=1)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                outputs = self.model.internlm_model(
                    inputs_embeds=prompt_embs,
                    attention_mask=atts_mask,
                    return_dict=True,
                    labels=None,
                )
        outputs = outputs.logits
        shift_logits = outputs[..., len_prompt:-1, :].contiguous()
        shift_labels = tars[..., 1+len_prompt:].contiguous()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.model.tokenizer.pad_token_id)
        sf = torch.nn.functional.softmax(shift_logits, dim=-1)

        loss = loss_fct(shift_logits, shift_labels.view(-1)).reshape(shift_labels.shape[0], shift_labels.shape[1])
        mask_length = None

        lens = (shift_labels !=
                self.model.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens

        return ce_loss

    def forward(
        self,
        x
    ):
        data_path, question, choices = x['data_path'], x['question'], x['choices']
        if "<img>" not in question:
            text = self.process_qa(question, choices)
            result = self.generate_answer_with_ppl(text, data_path)
        else: 
            text = self.process_interleved_qa(question, choices)
            result = self.generate_interleved_answer_with_ppl(text, data_path)

        return result


def build():
    return MLLM_Tester()
