import json
import os

import torchvision.models.video.resnet
from imageio import imsave
import torch
import torch.utils.data
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
import tqdm
from SegModel import FDTNet
import Config as config
import re
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, precision_score, \
    accuracy_score, jaccard_score, roc_curve, auc,matthews_corrcoef
import copy


root_path = './info'
checkpoint_path = os.path.join(root_path, 'best_model.pt')
img_save_path = "./img_save"

# hyperparameter
threshold = 0.5
batch_size = 50

if __name__ == "__main__":
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model = FDTNet()
    model.cuda()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()


    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    val_dataset = ImageToImage2D(config.val_dataset, val_tf, image_size=config.img_size, n_labels=2)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            num_workers=1,
                            persistent_workers=True,
                            pin_memory=True)

    unbinaryzation_dice_list = []
    unbinaryzation_uncertain_list = []
    binaryzation_dice_list = []
    binaryzation_dice_list1 = []
    binaryzation_dice_list2 = []

    key_press = None
    cnt = 0



    all_labels1 = []
    all_outputs1 = []

    all_labels2 = []
    all_outputs2 = []

    mcc_values1 = []
    mcc_values2 = []
    dice_values1 = []
    dice_values2 = []

    for img in tqdm.tqdm(val_loader):
        img = [torch.unsqueeze(img[0]["label"], dim=1), img[0]["image"]]
        temp = copy.deepcopy(img[0])
        label1 = torch.where(temp < 30, 1, 0)
        label2 = torch.where(temp < 230, 1, 0)
        img[0] = torch.cat([label1, label2], dim=1).float()

        if re.search("cuda", str(next(model.parameters()).device)):
            img = [item.cuda() for item in img]

        with torch.no_grad():
            prediction = model(img[1])
            prediction = prediction[:, :2, :, :]

        prediction[prediction >= threshold] = 1
        prediction[prediction <= threshold] = 0
        output1 = prediction[:, :1, :, :]
        output2 = prediction[:, 1:2, :, :]
        output2 = 1 - output2
        label1_flat = label1[0].flatten()
        output1_flat = output1[0].flatten().cpu()
        label2_flat = label2[0].flatten()
        output2_flat = output2[0].flatten().cpu()
        dice1 = f1_score(label1_flat, output1_flat)
        dice2 = f1_score(label2_flat, output2_flat)
        label3_flat = label1[1].flatten()
        output3_flat = output1[1].flatten().cpu()
        label4_flat = label2[1].flatten()
        output4_flat = output2[1].flatten().cpu()
        dice3 = f1_score(label3_flat, output3_flat)
        dice4 = f1_score(label4_flat, output4_flat)
        print(f"Average OC1 Dice: {dice1:.4f}")
        print(f"Average OD1 Dice: {dice2:.4f}")
        print(f"Average OC2 Dice: {dice3:.4f}")
        print(f"Average OD2 Dice: {dice4:.4f}")

