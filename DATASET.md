# Data Preparation

The question-answering data of SEED-Bench can be downloaded on Huggingface.

The structure of the JSON file for QA data is as follows:

```python
{
    "questions": [
        # Data of Evaluation Dimension 1-9
        {
            "answer": "A",                          # The answer is one of 'A', 'B', 'C', 'D'.
            "choice_a": "One",
            "choice_b": "Two",
            "choice_c": "Three",
            "choice_d": "Four",
            "data_id": "1454426_2591111986",        # The image file name in CC3M data provided by us.
            "data_type": "image",
            "question": "How many towels are in the image?",
            "question_id": "101669",                # Unique ID for each question
            "question_type_id": 5                   # Evaluation Dimension
        },
        ...
        # Data of Evaluation Dimension 10
        {
            "answer": "B",
            "choice_a": "throwing something in the air and letting it fall",
            "choice_b": "throwing something in the air and catching it",
            "choice_c": "lifting up one end of something, then letting it drop down",
            "choice_d": "poking something so that it falls over",
            "data_id": "94328.webm",                # The video file name in the original Something-Something v2 dataset.
            "data_type": "video",
            "question": "What is the action being carried out in the video?",
            "question_id": "v1738",
            "question_type_id": 10
        },
        ...
        # Data of Evaluation Dimension 11
        {
            "answer": "B",
            "choice_a": "fill saucepan",
            "choice_b": "move saucepan",
            "choice_c": "lift saucepan",
            "choice_d": "pour saucepan",
            "data_id": "P03/P03_24.MP4",            # The relative path of the video in the original Epic-kitchen dataset.
            "data_type": "video",
            "question": "Please anticipate what action will occur following the end of this video.",
            "question_id": "v1962",
            "question_type_id": 11,
            "segment": [648.0, 658.0]               # The start and end time (in seconds) of the input video segment
        },
        ...
        # Data of Evaluation Dimension 12
        {
            "answer": "C",
            "choice_a": "fetch bowl, get to cabinet, get to bowl, get to cereal, open cereal",
            "choice_b": "hold bowl, approach cabinet, acquire bowl, acquire cereal, expose cereal",
            "choice_c": "carry bowl, reach cabinet, reach bowl, reach cereal, open cereal",
            "choice_d": "grasp bowl, reach cabinet, approach bowl, approach cereal, open cereal",
            "data_id": "P06/cam01/P06_cereals.avi", # The relative path of the video in the original Breakfast dataset.
            "data_type": "video",
            "question": "Can you identify the sequence of actions in this video and list them in order?",
            "question_id": "v4397",
            "question_type_id": 12,
            "segment": [18, 166]                    # The start and end frame of the input video segment (15 fps).
        },
        ...
    ],
    "question_type": {
        "Scene Understanding": 1,
        "Instance Identity": 2,
        "Instance Location": 3,
        "Instance Attributes": 4,
        "Instances Counting": 5,
        "Spatial Relation": 6,
        "Instance Interaction": 7,
        "Visual Reasoning": 8,
        "Text Understanding": 9,
        "Action Recognition": 10,
        "Action Prediction": 11,
        "Procedure Understanding": 12
    }
}
```

The image data of evaluation dimension 1-9 comes from CC3M dataset and is uploaded to the HuggingFace repo [SEED-Bench](https://huggingface.co/datasets/AILab-CVC/SEED-Bench). 
The video data of evaluation dimension 10-12 comes from Something-Something v2, Epic-kitchen 100 and Breakfast dataset.

- Evaluation Dimension 1-9
    1. Include Scene Understanding, Instance Identity, Instance Location, Instance Attributes, Instances Counting, Spatial Relation, Instance Interaction, Visual Reasoning and Text Understanding.
    2. Image data can be downloaded from our HuggingFace repo.
    3. We use the image file name of each QA pair as 'data_id'.

- Evaluation Dimension 10-Action Recognition (Something-Something v2):
    1. We use 1740 videos from Something-Something v2 validation dataset. We use the video file name as 'data_id' of each QA pair.
    2. For downloading the videos, please refer to this [link](https://developer.qualcomm.com/software/ai-datasets/something-something).

- Evaluation Dimension 11-Action Prediction (Epic-kitchen 100):
    1. We use 138 long videos from Epic-kitchen validation dataset. We use the relative path of the video as 'data_id' of each QA pair. The start and end time (in seconds) of the input video segment is provided in the 'segment'.
    2. For downloading the videos, please refer to the download scripts provided in this [repo](https://github.com/epic-kitchens/epic-kitchens-download-scripts). We recommend downloading only the 138 videos required for the task 11 in the validation set, as each video is quite large.

- Evaluation Dimension 12-Procedure Understanding (Breakfast):
    1. We use videos from Breakfast dataset. We use the relative path of the video as 'data_id' of each QA pair. The start and end frame (integer) of the input video segment is provided in the 'segment' (The frame rate of each video is 15 fps).
    2. Please download videos from this [link](https://drive.google.com/open?id=1jgSoof1AatiDRpGY091qd4TEKF-BUt6I). ([The official page](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/))
    3. In our implementation, one of the videos (`P35/stereo/P35_milk_ch1.avi`) may not be decoded correctly. Using following commands with ffmpeg can fix this problem:
    ```shell
    ffmpeg -i P35/stereo/P35_milk_ch1.avi -c:v copy -c:a copy P35/stereo/P35_milk_ch1.mp4
    ffmpeg -i P35/stereo/P35_milk_ch1.mp4 P35/stereo/P35_milk_ch1.avi 
    ```

Please change the root data directory of dimension 1-12 (`cc3m_dir`, `dimension10_dir`, `dimension11_dir`, `dimension12_dir`) in [eval.py](eval.py).
