import copy
import json
import os
import re
import torch
import torch.nn as nn
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
import Config as config
from torch.utils.data import DataLoader
import time
import tqdm
import csv
from SegModel import FDTNet
from loss import dice_loss, PromptLoss
from torchvision import transforms
from criterions import dce_eviloss as UGR

# train hyperparameter.
iteration = 300
lr_base = 1e-4
para_save_per_n_epoch = 50
linear_epoch = 50
dice_epsilon = 1e-0




# training path
aisd_cur_path = os.path.join(".", "cur.json")
root_path = './info'
loss_path = os.path.join(root_path, 'loss.csv')
checkpoint_path = os.path.join(root_path, 'checkpoint.pt')
val_path = os.path.join(root_path, 'val_.csv')
prompt_path = os.path.join(root_path, 'prompt.pt')
best_model_path = os.path.join(root_path, 'best_model.pt')



def sche(optimizer):
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, verbose=True,
                                                   total_iters=linear_epoch)
    scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer,
                                                     verbose=True,
                                                     factor=1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                      [scheduler1, scheduler2],
                                                      milestones=[linear_epoch])
    return scheduler


if __name__ == "__main__":
    if os.path.exists(root_path) is False:
        os.mkdir(root_path)

    model = FDTNet()
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_base)
    scheduler = sche(optimizer)

    try:
        prompt = torch.load(checkpoint_path)
        best_model_info = torch.load(best_model_path)
    except FileNotFoundError:
        print("checkpoint not found.")
        epoch = 0
        best_dice = 0
    else:
        print("checkpoint load.")
        model.load_state_dict(prompt["model_state"])
        epoch = prompt["epoch"] + 1
        optimizer.load_state_dict(prompt["optim"])
        scheduler.load_state_dict(prompt["scheduler"])
        scheduler.step()
        best_dice = best_model_info["best_dice"]

    train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    train_dataset = ImageToImage2D(config.train_dataset, train_tf, image_size=config.img_size, n_labels=2)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf, image_size=config.img_size, n_labels=2)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size, num_workers=1,
                              persistent_workers=True,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            num_workers=1,
                            persistent_workers=True,
                            pin_memory=True)

    bce = nn.BCELoss()
    auxiliary_loss = PromptLoss()
    lambda_epochs = 150  # The annealing factor is set to half of the total epoch
    for epo in range(epoch, iteration):
        model.train()
        print("epoch " + str(epo) + " start.")
        loss_log = [time.asctime(time.localtime()), epo]
        global_step = epo
        for img in tqdm.tqdm(train_loader):
            img = [torch.unsqueeze(img[0]["label"], dim=1), img[0]["image"]]
            temp = copy.deepcopy(img[0])
            # Processing labels
            label1 = torch.where(temp < 30, 1, 0)
            label2 = torch.where(temp > 230, 1, 0)
            img[0] = torch.cat([label1, label2], dim=1).float()
            if re.search("cuda", str(next(model.parameters()).device)):
                img = [item.cuda() for item in img]

            evidence = model(img[1])
            # Input UGR module to generate predictions and losses
            evidence = evidence[:, :2, :, :]
            alpha = evidence + 1
            loss0 = UGR(img[0].to(torch.int64), alpha, 2, global_step, lambda_epochs)
            loss0 = torch.mean(loss0)
            loss = loss0
            loss_list = [loss0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list = [float(item.to("cpu").detach().numpy()) for item in loss_list]
            loss_log.append(loss_list)

        model.eval()
        validation_dice_list = []
        dice_list1 = []
        dice_list2 = []
        dice_list = []
        for img in tqdm.tqdm(val_loader):
            img = [torch.unsqueeze(img[0]["label"], dim=1), img[0]["image"]]
            temp = copy.deepcopy(img[0])
            # Processing labels
            label1 = torch.where(temp < 30, 1, 0)
            label2 = torch.where(temp > 230, 1, 0)
            img[0] = torch.cat([label1, label2], dim=1).float()
            if re.search("cuda", str(next(model.parameters()).device)):
                img = [item.cuda() for item in img]

            with torch.no_grad():
                prediction = model(img[1])
                prediction = prediction[:, :2, :, :]
                out0 = prediction[:, :1, :, :]
                out1 = prediction[:, 1:2, :, :]
                output0 = prediction[:, :2, :, :]
                lab1 = img[0][:, :1, :, :]
                lab2 = img[0][:, 1:2, :, :]
                dice1 = 1 - dice_loss(out0, lab1, avg=False, epsilon=dice_epsilon)
                dice2 = 1 - dice_loss(out1, lab2, avg=False, epsilon=dice_epsilon)
                dice = 1 - dice_loss(output0, img[0], avg=False, epsilon=dice_epsilon)


            dice_list1 = dice_list1 + dice1.tolist()
            dice_list2 = dice_list2 + dice2.tolist()
            dice_list = dice_list + dice.tolist()


        dice_avg1 = sum(dice_list1) / len(dice_list1)
        print("validation dice average 1", dice_avg1)
        dice_avg2 = sum(dice_list2) / len(dice_list2)
        print("validation dice average 2", dice_avg2)
        dice_avg = sum(dice_list) / len(dice_list)
        print("validation dice average", dice_avg)

        if dice_avg > best_dice:
            best_dice = dice_avg
            best_model_info = {
                "best_dice": best_dice,
                "model_state": model.state_dict(),
                "epoch": epo
            }
            torch.save(best_model_info, best_model_path)
        training_info = {
            "model_state": model.state_dict(),
            "epoch": epo,
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }

        torch.save(training_info, checkpoint_path)
        if epo % para_save_per_n_epoch == 0:
            if os.path.exists("./middle_state_dict") is False:
                os.mkdir("./middle_state_dict")
            torch.save(training_info, "./middle_state_dict/epoch" + str(epo) + ".pt")
        training_info = None


        with open(loss_path, "a") as lp:
            loss_write = csv.writer(lp)
            loss_write.writerow(loss_log)

        validation_dice_list.append([dice_avg])
        with open(val_path, "a") as val_p:
            val_write = csv.writer(val_p)
            val_write.writerow(validation_dice_list)

        print("epoch " + str(epo) + " finish.")
        scheduler.step()
