from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu
import av

from lavis.models import load_model_and_preprocess
from transformers.modeling_outputs import BaseModelOutput
from visual_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop,
    Stack, ToTorchFormatTensor
)
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import pdb


class MLLM_Tester(nn.Module):

    def __init__(self):
        super().__init__()
        # initialization of InstructBLIP
        self.model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5_instruct",
            model_type="flant5xl",
            is_eval=True
        )
        self.vis_processor = vis_processors['eval']

        crop_size = 224
        scale_size = 224
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]

        self.video_processor = transforms.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std)
        ])

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
        data_path, question, choices = x['data_path'], x['question'], x['choices']
        data_type = x['data_type']
        if data_type == 'image':
            # preprocessing images in evaluation dimension 1-9
            raw_image = Image.open(open(data_path, "rb")).convert("RGB")
            image = self.vis_processor(raw_image).unsqueeze(0).cuda()
            # pdb.set_trace()
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

        bs = image.size(0)
        n_segments = 1
        # prepare prompt based on the input question
        prompt = self.process_prompt(question)
        prompt = [prompt] * bs

        query_tokens = self.model.query_tokens.expand(bs, -1, -1)
        if self.model.qformer_text_input:
            text_Qformer = self.model.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.model.max_txt_len,
                return_tensors="pt"
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        if image.dim() == 5:
            inputs_t5, atts_t5 = [], []
            for j in range(image.size(2)):
                this_frame = image[:, :, j, :, :]
                with self.model.maybe_autocast():
                    frame_embeds = self.model.ln_vision(self.model.visual_encoder(this_frame))
                    frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.model.qformer_text_input:
                    frame_query_output = self.model.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.model.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )

                frame_inputs_t5 = self.model.t5_proj(frame_query_output.last_hidden_state[:, :query_tokens.size(1), :])
                frame_atts_t5 = torch.ones(frame_inputs_t5.size()[:-1], dtype=torch.long).to(image.device)
                inputs_t5.append(frame_inputs_t5)
                atts_t5.append(frame_atts_t5)
            inputs_t5 = torch.cat(inputs_t5, dim=1)
            atts_t5 = torch.cat(atts_t5, dim=1)
        else:
            with self.model.maybe_autocast():
                image_embeds = self.model.ln_vision(self.model.visual_encoder(image))  # [B, C, L]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.model.qformer_text_input:
                query_output = self.model.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_t5 = self.model.t5_proj(query_output.last_hidden_state[:, :query_tokens.size(1), :])
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        input_tokens = self.model.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)
        output_tokens = self.model.t5_tokenizer(
            choices, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        n_cands = len(choices)

        with self.model.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            encoder_outputs = self.model.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
            )

            losses = []
            for n in range(n_segments):
                seg_len = n_cands // n_segments
                if n == (n_segments - 1):
                    seg_len = n_cands - seg_len * (n_segments - 1)

                # this_encoder_outputs = copy.deepcopy(encoder_outputs)
                this_encoder_outputs = BaseModelOutput(
                    last_hidden_state=encoder_outputs[0].clone(),
                )

                this_encoder_outputs['last_hidden_state'] = this_encoder_outputs[0].repeat_interleave(seg_len, dim=0)
                this_encoder_atts = encoder_atts.repeat_interleave(seg_len, dim=0)

                start_i = n * (n_cands // n_segments)
                end_i = start_i + seg_len
                this_output_tokens_ids = output_tokens.input_ids[start_i:end_i].repeat(bs, 1)
                this_output_tokens_atts = output_tokens.attention_mask[start_i:end_i].repeat(bs, 1)

                this_targets = this_output_tokens_ids.masked_fill(
                    this_output_tokens_ids == self.model.t5_tokenizer.pad_token_id,
                    -100)

                outputs = self.model.t5_model(
                    encoder_outputs=this_encoder_outputs,
                    attention_mask=this_encoder_atts,
                    decoder_attention_mask=this_output_tokens_atts,
                    return_dict=True,
                    labels=this_targets,
                    reduction="none",
                )
                loss = outputs.loss

                loss = loss.reshape(bs, seg_len)
                losses.append(loss)
        # losses of 4 choices
        losses = torch.cat(losses, dim=-1)[0]
        return losses


def build():
    return MLLM_Tester()
