import os
import pdb
import model
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
from decord import VideoReader, cpu
import av

from visual_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop,
    Stack, ToTorchFormatTensor
)
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import InterpolationMode

def normalize():
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Normalize(mean, std)

class MLLM_Tester(nn.Module):
    def __init__(self, model_name='instructblip'):
        super().__init__()
        crop_size = 224
        scale_size = 224
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]

        self.vis_processor = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            normalize(),
        ]
    )
        self.video_processor = transforms.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std)
        ])
        
        self.model_info = model.get_model_info(model_name)
        
        print("Loading model & tokenizer")
        self.tokenizer = self.model_info['tokenizer'].from_pretrained(self.model_info['weight_name'])
        self.model = self.model_info['model'].from_pretrained(self.model_info['weight_name']).cuda()
    
    def process_prompt(self, question):
        return f"""Question: {question}\nAnswer:"""

    def transform_video(self, buffer):
        try:
            buffer = buffer.numpy()
        except AttributeError:
            try:
                buffer = buffer.asnumpy()
            except AttributeError:
                print("Both buffer.numpy() and buffer.asnumpy() failed.")
                buffer = None
        images_group = list()
        for fid in range(len(buffer)):
            images_group.append(Image.fromarray(buffer[fid]))
        torch_imgs = self.video_processor(images_group)
        return torch_imgs

    def get_index(self, num_frames, num_segments):
        if num_segments > num_frames:
            offsets = np.array([
                idx for idx in range(num_frames)
            ])
        else:
            # uniform sampling
            seg_size = float(num_frames - 1) / num_segments
            start = int(seg_size / 2)
            offsets = np.array([
                start + int(np.round(seg_size * idx)) for idx in range(num_segments)
            ])
        return offsets


    def forward(self, x):
        # data processing
        data_path, question, choices = x['data_path'], x['question'], x['choices']
        data_type = x['data_type']
        
        if data_type == 'image':
            # preprocessing images in evaluation dimension 1-9
            raw_image = Image.open(open(data_path, "rb")).convert("RGB")
            image = self.vis_processor(raw_image).unsqueeze(0).cuda()
        else:
            # preprocessing videos in evaluation dimension 10-12
            use_pyav = False
            if 'segment' in x.keys():
                segment = x['segment']
                if isinstance(segment[0], int):
                    # using pyav for decoding videos in evaluation dimension 12
                    use_pyav = True
                start, end = segment[0], segment[1]
            else:
                start = 0.0
                end = 0.0

            if use_pyav:
                # using pyav for decoding videos in evaluation dimension 12
                reader = av.open(data_path)
                frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
                video_len = len(frames)
                start_frame, end_frame = start, end
                end_frame = min(end_frame, video_len)
                offset = self.get_index(end_frame - start_frame, 8)
                frame_indices = offset + start_frame
                buffer = torch.stack([frames[idx] for idx in frame_indices])
            else:
                # using decord for decoding videos in evaluation dimension 10-11
                vr = VideoReader(data_path, num_threads=1, ctx=cpu(0))
                video_len = len(vr)
                fps = vr.get_avg_fps()
                if 'segment' in x.keys():
                    # obtain start and end frame for the video segment in evaluation dimension 11
                    start_frame = int(min(max(start * fps, 0), video_len - 1))
                    end_frame = int(min(max(end * fps, 0), video_len - 1))
                    tot_frames = int(end_frame - start_frame)
                    offset = self.get_index(tot_frames, 8)
                    frame_indices = offset + start_frame
                else:
                    # sample frames of the video in evaluation dimension 10
                    frame_indices = self.get_index(video_len - 1, 8)
                vr.seek(0)
                buffer = vr.get_batch(frame_indices)
            image = self.transform_video(buffer)

            TC, H, W = image.shape
            image = image.reshape(1, TC // 3, 3, H, W).permute((0, 2, 1, 3, 4)).cuda()

        # model processing

        bs = image.size(0)
        n_segments = 1

        prompt = self.process_prompt(question)
        prompt = [prompt] * bs
        
        losses = []

        input_tokenized = self.tokenizer(text = prompt, return_tensors='pt', padding="longest").to(image.device)

        query_tokens = self.model.query_tokens.expand(bs, -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts,input_tokenized.qformer_attention_mask], dim=1)

        output_attentions, output_hidden_states, return_dict = None, None, None
        if data_type == 'image':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                image_embeds = self.model.vision_model(pixel_values = image)[0]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            
            query_output = self.model.qformer(
                input_ids=input_tokenized.qformer_input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_t5 = self.model.language_projection(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        else:
            # preprocessing videos in evaluation dimension 10-12
            inputs_t5, atts_t5 = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    frame_embeds = self.model.vision_model(pixel_values = this_frame)[0]
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                frame_query_output = self.model.qformer(
                    input_ids=input_tokenized.qformer_input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_atts,
                    return_dict=True,
                )

                frame_inputs_t5 = self.model.language_projection(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long, device=image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)

        encoder_atts = torch.cat([atts_t5, input_tokenized.attention_mask], dim=1)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            inputs_embeds = self.model.get_input_embeddings()(input_tokenized.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            for choice in choices:
                output_tokenized = self.tokenizer.tokenizer(text=choice, return_tensors='pt', padding="longest", truncation=True).to(image.device)
                targets = output_tokenized.input_ids.masked_fill(
                output_tokenized.input_ids == self.tokenizer.tokenizer.pad_token_id, -100
                )
                this_encoder_atts = output_tokenized.attention_mask

                outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                return_dict=True,
                labels=targets,
                )
                
                loss = outputs.loss if return_dict else outputs[0]
                losses.append(loss)
                
        return torch.stack(losses)

def build():
    return MLLM_Tester()