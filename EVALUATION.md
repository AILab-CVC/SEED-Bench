## Run Evaluation for SEED-Bench-2

The evaluation metric is provided in [eval.py]([eval.py](https://github.com/AILab-CVC/SEED-Bench/blob/main/SEED-Bench-2/eval.py)). We use [InternLM_Xcomposer_VL](https://arxiv.org/pdf/2309.15112.pdf) as an example. To run the following evaluation code, please refer to [repo](https://github.com/salesforce/LAVIS) for the environment preparation.

```shell
python eval.py --model InternLM_Xcomposer_VL --anno_path SEED-Bench_v2_level1_2_3.json --output-dir results --evaluate_level L2 --evaluate_part all --evaluate_version v2
```

After the evaluation is finished, you can obtain the accuracy of each evaluation dimension and also 'results.json' in 'results' folder.

If you want to evaluate your own models, please provide the interface like [iInternLM_Xcomposer_VL_interface.py](https://github.com/AILab-CVC/SEED-Bench/blob/main/SEED-Bench-2/model/InternLM_Xcomposer_VL_interface.py).


## Run Evaluation for SEED-Bench-1

The evaluation metric is provided in [eval.py](eval.py). We use [InstructBLIP](https://arxiv.org/abs/2305.06500) as an example. To run the following evaluation code, please refer to [repo](https://github.com/salesforce/LAVIS) for the environment preparation.

```shell
python eval.py --model instruct_blip --anno_path SEED-Bench.json --output-dir results --task all
```

After the evaluation is finished, you can obtain the accuracy of each evaluation dimension and also 'results.json' in 'results' folder, which can be submitted to [SEED-Bench Leaderboard](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard).

If you want to evaluate your own models, please provide the interface like [instruct_blip_interface.py](https://github.com/AILab-CVC/SEED-Bench/blob/main/model/instruct_blip_interface.py).

Note that to evaluate models with multiple-choice questions, we adopt the answer ranking strategy
following GPT-3. Specifically, for each choice of a question, we compute the likelihood 
that a model generates the content of this choice given the question. 
We select the choice with the highest likelihood as model's prediction. 
Our evaluation strategy does not rely on the instruction-following capabilities 
of models to output 'A' or 'B' or 'C' or 'D'.
