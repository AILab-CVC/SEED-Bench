import os
import json
import argparse

import torch
from tqdm import tqdm
import numpy as np
import random
import pdb

# root directory of cc3m
cc3m_dir = "/YOUR_PATH_TO/seed_bench_image"
# root directory of seed bench v2
seed_bench_v2_dir = "/YOUR_PATH_TO/seed_bench_image_v2"

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def filter_questions(data, level='L2', subpart='all', version='v2'):
    if level == "L1":
        valid_level_data = ['L1']
    elif level == "L2":
        valid_level_data = ['L1', 'L2']
    elif task == "L3":
        valid_level_data = ['L1', 'L2', 'L3']
    else:
        raise ValueError(f"Invalid level: {level}")
    data = [q for q in data if q["level"] in valid_level_data]

    if subpart in ['Single-Image & Text Comprehension', 'Multiple-Images & Text Comprehension', 'Video & Text Comprehension', 'Interleaved Image & Text Comprehension', 'Image Generation', 'Image & Text Generation']:
        valid_subgroup_data = subgroup
    elif subpart == 'all':
        valid_subgroup_data = ['Single-Image & Text Comprehension', 'Multiple-Images & Text Comprehension', 'Video & Text Comprehension', 'Interleaved Image & Text Comprehension', 'Image Generation', 'Image & Text Generation']
    else:
        raise ValueError(f"Invalid subpart: {subpart}")
    data = [q for q in data if q["subpart"] in valid_subgroup_data]

    if version == 'v1':
        valid_version_data = ['v1']
    elif version == 'v2':
        valid_version_data = ['v1', 'v2']
    else:
        raise ValueError(f"Invalid version: {version}")
    data = [q for q in data if q["version"] in valid_version_data]

    return data

def build_model(model_name):
    if model_name == 'InternLM_Xcomposer_VL':
        from model.InternLM_Xcomposer_VL_interface import build
    elif model_name == 'llava_1.5':
        from model.llava_v2_interface import build 
    model = build()

    return model

def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def run_inference(model, qa_anno, output_dir):
    total_qa_num = len(qa_anno)
    answer_list = []
    output_f = open(os.path.join(output_dir, "results.json"), "a")
    step = 0
    for qa_item in tqdm(qa_anno):

        data_info = {
            'question': qa_item['question'],
            'choices': [qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']],
        }

        if qa_item["data_source"] == 'cc3m':
            image_dir = cc3m_dir
        elif qa_item["data_source"] == 'SEED-Bench v2':
            image_dir = seed_bench_v2_dir
        else:
            raise ValueError("The data type is not valid.")
        if type(qa_item['data_id']) is list:
            data_path = [os.path.join(image_dir, path) for path in qa_item['data_id']]
        else:
            data_path = os.path.join(image_dir, qa_item['data_id'])
        data_info['data_path'] = data_path

        # losses: loss values of 4 choices, shape=[4]
        with torch.no_grad():
            losses = model(data_info)
        class_ranks = np.argsort(losses)
        pred_id = ['A', 'B', 'C', 'D'][class_ranks[0]]
        gt = qa_item['answer']
        answer_record = {
            'question_id': qa_item['question_id'],
            'prediction': pred_id
        }
        answer_list.append(answer_record)
        # output prediction record for each question
        output_f.write(json.dumps(answer_record) + "\n")
        step += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='InternLM_Xcomposer_VL')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench_v2_level1_2_3.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--evaluate_level', type=str, default='L2')
    parser.add_argument('--evaluate_part', type=str, default='all')
    parser.add_argument('--evaluate_version', type=str, default='v2')
    args = parser.parse_args()
    
    args = parser.parse_args()

    qa_anno = json.load(open(args.anno_path, 'rb'))
    if 'questions' in qa_anno.keys():
        qa_anno = qa_anno['questions']
    qa_anno = filter_questions(qa_anno, args.evaluate_level, args.evaluate_part, args.evaluate_version)
    pdb.set_trace()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print(f'evaluating.. {args.model}')
    # The interface for testing MLLMs
    model = build_model(args.model).cuda()
    run_inference(model, qa_anno, args.output_dir)
