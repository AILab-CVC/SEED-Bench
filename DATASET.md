# Data Preparation for SEED-Bench-2

The question-answering data of SEED-Bench-2 can be downloaded on Huggingface.

The structure of the JSON file for QA data is as follows:

```python
{
    "questions": [
        # Data of Evaluation Dimension 1-16
        {
            "answer": "A",                                      # The answer is one of 'A', 'B', 'C', 'D'.
            "choice_a": "One",
            "choice_b": "Two",
            "choice_c": "Three",
            "choice_d": "Four",
            "data_id": "1454426_2591111986",                    # The image file name in CC3M data provided by us.
            "data_source": "cc3m",                              # The data source of image.
            "data_type": "Single Image",                        # The type for data.
            "level": "L1",                                      # The question's level.
            "question": "How many towels are in the image?",
            "question_id": "5_0",                               # Unique ID for each question.
            "question_type_id": 5,                              # Evaluation Dimension.
            "subpart": "Single-Image & Text Comprehension",     # The question's subpart.
            "version": "v1"                                     # The question's version.
        },
        ...
        # Data of Evaluation Dimension 17-18
        {
            "answer": "B",
            "choice_a": "The person in the blue shirt from the image on the left is missing, and the dark grey SUV is replaced with a red car in the photo on the right.",
            "choice_b": "The person in the blue shirt from the image on the left is no longer present in the right image, and the dark grey SUV is gone in the photo on the right.",
            "choice_c": "There is an extra tree in the image on the right, and the dark grey SUV is replaced with a red car in the photo on the right.",
            "choice_d": "The person in the blue shirt from the image on the left is wearing a different colored shirt in the right image, and the dark grey SUV is replaced with a red car in the photo on the right.",
            "data_id": [                                        # The image file names list in SEED-Bench-2 data provided by us.
                "task17/SD_image/SD_IMG_00235_1.png",
                "task17/SD_image/SD_IMG_00235_2.png"
            ],
            "data_source": "SEED-Bench v2",
            "data_type": "Multiple Images",
            "level": "L1",
            "question": "What are the differences between the two images?",
            "question_id": "17_0",
            "question_type_id": 17,
            "subpart": "Multiple-Images & Text Comprehension",
            "version": "v2"
        },
        ...
        # Data of Evaluation Dimension 19-22
        {
            "answer": "B",
            "choice_a": "Hung a white robe up on the wall",
            "choice_b": "Cut a knife on the table",
            "choice_c": "Placed a trash can next to the door",
            "choice_d": "Hung a blue shirt and tie up on the wall",
            "data_id": [                                        # The frame file names list in SEED-Bench-2 data provided by us.
                "task19/Charades_test_frame/WRQ95/1.png",
                "task19/Charades_test_frame/WRQ95/2.png",
                "task19/Charades_test_frame/WRQ95/3.png",
                "task19/Charades_test_frame/WRQ95/4.png",
                "task19/Charades_test_frame/WRQ95/5.png",
                "task19/Charades_test_frame/WRQ95/6.png",
                "task19/Charades_test_frame/WRQ95/7.png",
                "task19/Charades_test_frame/WRQ95/8.png"
            ],
            "data_source": "SEED-Bench v2",
            "data_type": "Video",
            "level": "L1",
            "question": "What activity did the person in the white shirt perform after he leaned over the wooden table?",
            "question_id": "19_0",
            "question_type_id": 19,
            "subpart": "Video & Text Comprehension",
            "version": "v2"
        },
        ...
        # Data of Evaluation Dimension 23-24
        {
            "answer": "B",
            "choice_a": "The man and woman in the image are both looking away from the camera.",
            "choice_b": "The woman's hair is black.",
            "choice_c": "The woman's dog is on the couch next to her in the image.",
            "choice_d": "There are two people in the image.",
            "data_id": [
                "task23/ICL_images/in_context_attribute_2/1.jpg",
                "task23/ICL_images/in_context_attribute_2/2.jpg",
                "task23/ICL_images/in_context_attribute_2/3.jpg"
            ],
            "data_source": "SEED-Bench v2",
            "data_type": "Interleaved Image",
            "level": "L2",
            "question": "<img>: The predominant color of the uniforms worn by the players is blue. <img>: The most notable color present in the woman's outfit is orange. <img>:",  # <img> indicates the position of an individual image within a sequence in data_id
            "question_id": "23_0",
            "question_type_id": 23,
            "subpart": "Interleaved Image & Text Comprehension",
            "version": "v2"
        },
        ...
        # Data of Evaluation Dimension 25
        {
            "answer": "A",
            "choice_a": "1.jpg",
            "choice_b": "2.jpg",
            "choice_c": "3.jpg",
            "choice_d": "4.jpg",
            "data_id": [                    # The image file names list of options in SEED-Bench-2 data provided by us.
                "task25/image/0/1.jpg",
                "task25/image/0/2.jpg",
                "task25/image/0/3.jpg",
                "task25/image/0/4.jpg"
            ],
            "data_source": "SEED-Bench v2",
            "data_type": "Image Generation",
            "level": "L3",
            "question": "Which picture below better fits the description: A brown purse is sitting on a green bench.",
            "question_id": "25_0",
            "question_type_id": 25,
            "subpart": "Image Generation",
            "version": "v2"
        },
        ...
        # Data of Evaluation Dimension 26
        {
            "answer": "B",
            "choice_a": "v2799/2.png",
            "choice_b": "v1752/2.png",
            "choice_c": "v3219/1.png",
            "choice_d": "v3134/2.png",
            "data_id": [                    # The list of image file names for questions and options consists of the first three files as questions, and the remaining files as option images.
                "task26/kitchen_start_end_frame/v1753/1.png",
                "task26/kitchen_start_end_frame/v1747/1.png",
                "task26/kitchen_start_end_frame/v1748/1.png",
                "task26/kitchen_start_end_frame/v2799/2.png",
                "task26/kitchen_start_end_frame/v1752/2.png",
                "task26/kitchen_start_end_frame/v3219/1.png",
                "task26/kitchen_start_end_frame/v3134/2.png"
            ],
            "data_source": "SEED-Bench v2",
            "data_type": "Image Generation",
            "level": "L3",
            "question": "What will happen next?",
            "question_id": "26_0",
            "question_type_id": 26,
            "subpart": "Image Generation",
            "version": "v2"
        },
        # Data of Evaluation Dimension 27
        ...
        {
            "answer": "A",
            "choice_a": "The Sydney Opera House is a multi-venue performing arts center in Sydney, Australia. It features a series of large, white, sail-like shells that form the roof, supported by a platform that juts out into the harbor. The shells are made of precast concrete panels, covered in over a million white and cream-colored Swedish tiles. The building is surrounded by water on three sides and sits on a peninsula, making it a prominent landmark in the Sydney skyline.",
            "choice_b": "The Sydney Opera House is a tall skyscraper with a rectangular shape.",
            "choice_c": "The Sydney Opera House is a large circular stadium with an open roof.",
            "choice_d": "The Sydney Opera House is a bridge with a large steel arch.",
            "data_id": [                    # The image file names list of options in SEED-Bench-2 data provided by us.
                "task27/original/1698047226_8596642.jpg",
                "task27/original/cdn.britannica.com_98_94398-050-FBE19E2C_Skyscrapers-Singapore.jpg",
                "task27/original/1698896712_4471433.jpg",
                "task27/original/www.bukaka.com_asset_uploads_images_JBT-Arc_Steel_Truss_Bridge2.jpg"
            ],
            "data_source": "SEED-Bench v2",
            "data_type": "Image & Text Generation",
            "level": "L3",
            "question": "What does the Sydney Opera House look like?",
            "question_id": "27_0",
            "question_type_id": 27,
            "subpart": "Image & Text Generation",
            "version": "v2"
        },
        ...
    ],
    "question_type": {
        "Scene Understanding": 1,
        "Instance Identity": 2,
        "Instance Attributes": 3,
        "Instance Location": 4,
        "Instances Counting": 5,
        "Spatial Relation": 6,
        "Instance Interaction": 7,
        "Visual Reasoning": 8,
        "Text Understanding": 9,
        "Celebrity Recognition": 10,
        "Landmark Recognition": 11,
        "Chart Understanding": 12,
        "Visual Referring Expression": 13,
        "Science Knowledge": 14,
        "Emotion Recognition": 15,
        "Visual Mathematics": 16,
        "Difference Spotting": 17,
        "Meme Comprehension": 18,
        "Global Video Understanding": 19,
        "Action Recognition": 20,
        "Action Prediction": 21,
        "Procedure Understanding": 22,
        "In-Context Captioning": 23,
        "Interleaved Image-Text Analysis": 24,
        "Text-to-Image Generation": 25,
        "Next Image Prediction": 26,
        "Text-Image Creation": 27
    }
}
```
Data for evaluation is uploaded to the HuggingFace repo [SEED-Bench-2](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2).

- Evaluation Dimension 1-9
    1. Include Scene Understanding, Instance Identity, Instance Attributes, Instance Location, Instances Counting, Spatial Relation, Instance Interaction, Visual Reasoning and Text Understanding.
    2. Image data can be downloaded from our HuggingFace repo [SEED-Bench-2](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2) as [cc3m-image.zip](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2/blob/main/cc3m-image.zip).
    3. We use the image file name of each QA pair as 'data_id'.

- Evaluation Other Dimension:
    1. Include Text Understanding, Celebrity Recognition, Landmark Recognition, Chart Understanding, Visual Referring Expression, Science Knowledge, Emotion Recognition, Visual Mathmatics, Difference Spotting, Meme Comprehension, Global Video Understanding, Action Recognition, Action Prediction, Procedure Understanding, In-Context Captioning, Interleaved Image-Text Analysis, Text-to-Image Generation, Next Image Prediction, Text-Image Creation.
    2. Data can be downloaded from our HuggingFace repo [SEED-Bench-2](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2) as [SEED-Bench-2-image.zip.***](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2).


Please update the root data directories for dimensions 1-27 (`cc3m_dir`, `seed_bench_v2_dir`) in [eval.py](https://github.com/AILab-CVC/SEED-Bench/blob/main/SEED-Bench-2/eval.py), setting `cc3m_dir` for [cc3m-image.zip](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2/blob/main/cc3m-image.zip)and `seed_bench_v2_dir` for [SEED-Bench-2-image.zip.***](https://huggingface.co/datasets/AILab-CVC/SEED-Bench-2).


# Data Preparation for SEED-Bench-1

The question-answering data of SEED-Bench-1 can be downloaded on Huggingface.

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
        "Instance Attributes": 3,
        "Instance Location": 4,
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
    1. Include Scene Understanding, Instance Identity, Instance Attributes, Instance Location, Instances Counting, Spatial Relation, Instance Interaction, Visual Reasoning and Text Understanding.
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
    
In addition, we also provide the video frames corresponding to the questions. For each question, we evenly sample the corresponding 8 frames for answering the questions as [v1_video.zip.***](https://huggingface.co/datasets/AILab-CVC/SEED-Bench).

Please change the root data directory of dimension 1-12 (`cc3m_dir`, `dimension10_dir`, `dimension11_dir`, `dimension12_dir`) in [eval.py](eval.py).
