# AICITY2023_Track5
This repository is the implementation of the solution for the AICITY2024 Track5 challenge - Detecting Violation of Helmet Rule for Motorcyclists.

![framework](./static/system_architecture_vec.svg)

# About us:
- Team name: Helios
- Team members:
  - [Vo Anh Hao](https://www.linkedin.com/in/haovo0602/)
  - [Tran Sieu](https://www.linkedin.com/in/sieutran102)
  - [Nguyen Minh Duc](https://www.linkedin.com/in/nguyễn-minh-đức-b5359124b)

# Public Leaderboard
| Rank | Team ID | Team Name    | Score  |
|------|---------|--------------|--------|
| 1    | 99      | **Helios (Our)**       | **0.4860** |
| 2    | 76      | CMSR_PANDA   | 0.4824 |
| 3    | 9       | VNPT AI      | 0.4792 |
| 4    | 155     | TeleAI       | 0.4675 |
| 5    | 5       | SKKU-AutoLab | 0.4644 |


# Inference
Setup environment:
```
cd sources/inference
docker build -t helios:latest .
cd ../../
```

Run docker:
```
docker run -v /path/to/data:/path/to/data -v /path/to/repo:/target/path --gpus all -it helios:latest bash
cd /target/path
```

Dowload models, [here](https://drive.google.com/drive/folders/1qHaUTpaTk7PwzpgdZ-NvgKfzeal01Va9?usp=sharing). Then place them in `sources/inference/weights` folder.

Run inference on test set:
```
bash scripts/inference.sh
```
Note that:
- `--batch_size` is the number of images to inference at the same time.
- `--checkpoint_path` is the path to the checkpoint file.
- `--config_path` is the path to the configuration file.
- `--test_path` is the path to the test set.

# Training
This is instruction to train the model.

## Prepare dataset
Dowload training dataset from website of [aicitychallenge](https://www.aicitychallenge.org/2024-data-and-evaluation/). Unzip and place the folder `aicity2024_track5_train` in `data` folder. The structure of the folder should be like this:
```
data
└── aicity2024_track5_train
    ├── videos
    │   ├── 001.mp4
    │   ├── 002.mp4
    │   └── ...
    ├── gt.txt
```

After that, run the following command to prepare the dataset:
```
bash scripts/convert.sh
```

## Train
Firstly, dowload pre-trained models [here](https://drive.google.com/file/d/1ffDz9lGNAjEF7iXzINZezZ4alx6S0KcO/view?usp=drive_link), then place it in folder `sources/Co-DETR/models`.
To start the training process, run:
```
cd sources/Co-DETR
docker compose up --build -d
```
Then we use checkpoints name `epoch_10.pth` and `epoch_15.pth` for testing.

# Acknowledgements
[Co-DETR](https://github.com/Sense-X/Co-DETR) The base code for training and it is strong for object detection task.
