# FDT-Net

1.Main Environment
python 3.10.9  pytorch 2.0.1 torchvision 0.15.2

2. Prepare the dataset
Scale your dataset into train, val and test. The file format reference is as follows
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png
  - test
    - images
      - .png
    - masks
      - .png

3.Train the FDT-Net
Config.py configures the corresponding parameters and paths
train.py training documents

4.test
validation.py
