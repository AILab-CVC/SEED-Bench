# SEED-Bench: Benchmarking Multimodal Large Language Models
[SEED-Bench-2 Arxiv](https://arxiv.org/abs/2311.17092)

[SEED-Bench-1 Arxiv](https://arxiv.org/abs/2307.16125)

 <img src="https://github.com/AILab-CVC/SEED-Bench/blob/main/figs/seed-bench-2.jpg" width = "600"  alt="图片名称" align=center />

 SEED-Bench-2 comprises 24K multiple-choice questions with accurate human annotations, which spans 27 dimensions, including the evaluation of both text and image generation.
 
 SEED-Bench-1 consists of 19K multiple-choice questions with accurate human annotations, covering 12 evaluation dimensions
including both the spatial and temporal understanding.
## News
**[2024.4.23]** We are pleased to share the comprehensive evaluation results for [Gemini-Vision-Pro](https://gemini.google.com/) and [Claude-3-Opus](https://www.anthropic.com/news/claude-3-family) on [SEED-Bench-1](https://arxiv.org/abs/2307.16125) and [SEED-Bench-2](https://arxiv.org/abs/2311.17092). You can access detailed performance on the [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard). Please note that for [Gemini-Vision-Pro](https://gemini.google.com/) we only report task performance when the model responds with at least 50% valid data in the task. 

**[2024.2.27]** [SEED-Bench](https://arxiv.org/abs/2311.17092) is accepted by **CVPR 2024**.

**[2023.12.18]** We have placed the comprehensive evaluation results for [GPT-4v](https://openai.com/research/gpt-4v-system-card) on [SEED-Bench-1](https://arxiv.org/abs/2307.16125) and [SEED-Bench-2](https://arxiv.org/abs/2311.17092). These can be accessed at [GPT-4V for SEED-Bench-1](https://github.com/AILab-CVC/SEED-Bench/blob/main/evaluate_result/SEED-Bench-1/GPT-4V.json) and [GPT-4V for SEED-Bench-2](https://github.com/AILab-CVC/SEED-Bench/blob/main/evaluate_result/SEED-Bench-2/GPT-4V.json). If you're interested, please feel free to take a look.

**[2023.12.4]** We have updated the [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) for [SEED-Bench-2](https://arxiv.org/abs/2311.17092). Additionally, we have updated the evaluation results for [GPT-4v](https://openai.com/research/gpt-4v-system-card) on both [SEED-Bench-1](https://arxiv.org/abs/2307.16125) and [SEED-Bench-2](https://arxiv.org/abs/2311.17092). If you are interested, please visit the [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) for more details.

**[2023.11.30]** We have updated the SEED-Bench-v1 JSON (manually screening the multiple-choice questions for videos) and provided corresponding video frames for easier testing. Please refer to [SEED-Bench](https://huggingface.co/datasets/AILab-CVC/SEED-Bench) for more information.

**[2023.11.27]** [SEED-Bench-2](https://arxiv.org/abs/2311.17092) is released! Data and evaluation code is available now.

**[2023.9.9]** We are actively looking for self-motivated interns. Please feel free to reach out if you are interested.

**[2023.8.16]** [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) is released! You can upload your model's results now.

**[2023.7.30]** [SEED-Bench](https://arxiv.org/abs/2307.16125) is released! Data and evaluation code is available now.

## Leaderboard
Welcome to [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard)!

### Leaderboard Submission

You can submit your model results in [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) now. You can use our evaluation code to obtain 'results.json' in 'results' folder as below.

```shell
python eval.py --model instruct_blip --anno_path SEED-Bench.json --output-dir results --task all
```

Then you can upload 'results.json' in [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard).

After submitting, please press refresh button to get the latest results.

## Data Preparation

You can download the data of SEED-Bench released on HuggingFace repo [SEED-Bench](https://huggingface.co/datasets/AILab-CVC/SEED-Bench) and [SEED-Bench-2](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2).
Please refer to [DATASET.md](DATASET.md) for data preparation.

## Installation

Please refer to [INSTALL.md](INSTALL.md).

## Run Evaluation

Please refer to [EVALUATION.md](EVALUATION.md).

## License
SEED-Bench is released under Apache License Version 2.0.

## Declaration

### SEED-Bench-2
Data Sources:
- Dimensions 1-9, 23 (In-Context Captioning): Conceptual Captions Dataset (https://ai.google.com/research/ConceptualCaptions/) under its license (https://github.com/google-research-datasets/conceptual-captions/blob/master/LICENSE). Copyright belongs to the original dataset owner.
- Dimension 9 (Text Recognition): ICDAR2003 (http://www.imglab.org/db/index.html), ICDAR2013(https://rrc.cvc.uab.es/?ch=2), IIIT5k(https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset), and SVT(http://vision.ucsd.edu/~kai/svt/). Copyright belongs to the original dataset owner.
- Dimension 10 (Celebrity Recognition): MME (https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) and MMBench (https://github.com/open-compass/MMBench) under MMBench license (https://github.com/open-compass/MMBench/blob/main/LICENSE). Copyright belongs to the original dataset owners.
- Dimension 11 (Landmark Recognition): Google Landmark Dataset v2 (https://github.com/cvdfoundation/google-landmark) under CC-BY licenses without ND restrictions.
- Dimension 12 (Chart Understanding): PlotQA (https://github.com/NiteshMethani/PlotQA) under its license (https://github.com/NiteshMethani/PlotQA/blob/master/LICENSE).
- Dimension 13 (Visual Referring Expression): VCR (http://visualcommonsense.com) under its license (http://visualcommonsense.com/license/).
- Dimension 14 (Science Knowledge): ScienceQA (https://github.com/lupantech/ScienceQA) under its license (https://github.com/lupantech/ScienceQA/blob/main/LICENSE-DATA).
- Dimension 15 (Emotion Recognition): FER2013 (https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data) under its license (https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/rules#7-competition-data).
- Dimension 16 (Visual Mathematics): MME (https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) and data from the internet under CC-BY licenses.
- Dimension 17 (Difference Spotting): MIMICIT (https://github.com/Luodian/Otter/blob/main/mimic-it/README.md) under its license (https://github.com/Luodian/Otter/tree/main/mimic-it#eggs).
- Dimension 18 (Meme Comprehension): Data from the internet under CC-BY licenses.
- Dimension 19 (Global Video Understanding): Charades (https://prior.allenai.org/projects/charades) under its license (https://prior.allenai.org/projects/data/charades/license.txt). SEED-Bench-2 provides 8 frames per video.
- Dimensions 20-22 (Action Recognition, Action Prediction, Procedure Understanding): Something-Something v2 (https://developer.qualcomm.com/software/ai-datasets/something-something), Epic-Kitchen 100 (https://epic-kitchens.github.io/2023), and Breakfast (https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/). SEED-Bench-2 provides 8 frames per video.
- Dimension 24 (Interleaved Image-Text Analysis): Data from the internet under CC-BY licenses.
- Dimension 25 (Text-to-Image Generation): CC-500 (https://github.com/weixi-feng/Structured-Diffusion-Guidance) and ABC-6k (https://github.com/weixi-feng/Structured-Diffusion-Guidance) under their license (https://github.com/weixi-feng/Structured-Diffusion-Guidance/blob/master/LICENSE), with images generated by Stable-Diffusion-XL (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) under its license (https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md).
- Dimension 26 (Next Image Prediction): Epic-Kitchen 100 (https://epic-kitchens.github.io/2023) under its license (https://creativecommons.org/licenses/by-nc/4.0/).
- Dimension 27 (Text-Image Creation): Data from the internet under CC-BY licenses.

Please contact us if you believe any data infringes upon your rights, and we will remove it.

### SEED-Bench-1
For the images of SEED-Bench-1, we use the data from Conceptual Captions Dataset (https://ai.google.com/research/ConceptualCaptions/)
following its license (https://github.com/google-research-datasets/conceptual-captions/blob/master/LICENSE).
Tencent does not hold the copyright for these images and the copyright belongs to the original owner of Conceptual Captions Dataset. 

For the videos of SEED-Bench-1, we use tha data from Something-Something v2 (https://developer.qualcomm.com/software/ai-datasets/something-something),
Epic-kitchen 100 (https://epic-kitchens.github.io/2023) and 
Breakfast (https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/). We only provide the video name. Please download them in their official websites.


## Citing
If you find this repository helpful, please consider citing it:
```
@article{li2023seed2,
  title={SEED-Bench-2: Benchmarking Multimodal Large Language Models},
  author={Li, Bohao and Ge, Yuying and Ge, Yixiao and Wang, Guangzhi and Wang, Rui and Zhang, Ruimao and Shan, Ying},
  journal={arXiv preprint arXiv:2311.17092},
  year={2023}
  }

@article{li2023seed,
  title={Seed-bench: Benchmarking multimodal llms with generative comprehension},
  author={Li, Bohao and Wang, Rui and Wang, Guangzhi and Ge, Yuying and Ge, Yixiao and Shan, Ying},
  journal={arXiv preprint arXiv:2307.16125},
  year={2023}
}
```
