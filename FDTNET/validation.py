import json
import os

import torchvision.models.video.resnet
from imageio import imsave
import torch
import torch.utils.data
from Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D
import numpy as np
import tqdm
from train import dice_loss, dice_epsilon, val_length, down_sample , down_sample_rate
import cv2
from transunet import TransUnet
import Config as config
import re
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, precision_score, \
    accuracy_score, jaccard_score, roc_curve, auc,matthews_corrcoef
import copy
from thop import profile
import matplotlib.pyplot as plt

root_path = './info'
checkpoint_path = os.path.join(root_path, 'best_model-0.3.pt')
#aisd_cur_path = "./cur.json"
img_save_path = "./img_save"

# hyperparameter
threshold = 0.5
batch_size = 50

def accuracy(y_true, y_pred):
    correct = torch.sum(y_true == y_pred).item()  # 计算预测正确的样本数
    total = len(y_true) # 总样本数
    accuracy = correct/ total
    return accuracy


def precision(y_true, y_pred):
    true_positives = torch.sum((y_true == 1) & (y_pred == 1)).item()
    false_positives = torch.sum((y_true == 0) & (y_pred == 1)).item()

    denominator = true_positives + false_positives

    # 避免分母为零，设置默认值
    precision = true_positives / denominator if denominator != 0 else 0.0

    return precision
if __name__ == "__main__":
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model = TransUnet()
    model.cuda()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    #val_cur = json.load(open(aisd_cur_path, "r"))
    #val_tf = ValGenerator(output_size=[Config.img_size, Config.img_size])
    #validation_set = ImageToImage2D(Config.test_dataset,val_tf,image_size=224,n_labels=2)
    #val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size)
    # tf_test = ValGenerator(output_size=[Config.img_size, Config.img_size])
    # test_dataset = ImageToImage2D(Config.test_dataset, tf_test, image_size=Config.img_size)
    # test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
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
    # prompt_dice_list = []
    precision1 = []
    recall1 = []
    accuracy1 = []
    f1_1 = []
    jaccard1 = []
    precision2 = []
    recall2 = []
    accuracy2 = []
    f1_2 = []
    jaccard2 = []
    key_press = None
    cnt = 0

    # Model


    dummy_input = torch.randn(2, 3, 512, 512).cuda()
    flops, params = profile(model, (dummy_input,))
    print('flops:', flops, 'params:', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

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
        #label2 = 1 - label2
        img[0] = torch.cat([label1, label2], dim=1).float()

        if re.search("cuda", str(next(model.parameters()).device)):
            img = [item.cuda() for item in img]

        with torch.no_grad():
            fe, fe2, log, edge,output = model(img[1])
            output = output[:, :2, :, :]

        # #计算MCC
        output[output > threshold] = 1
        output[output < threshold] = 0
        output1 = output[:, :1, :, :]
        output2 = output[:, 1:2, :, :]
        output2 = 1 - output2
        label1_flat = label1[0].flatten()
        output1_flat = output1[0].flatten().cpu()
        # mcc1 = matthews_corrcoef(label1_flat, output1_flat)
        # mcc_values1.append(mcc1)
        label2_flat = label2[0].flatten()
        output2_flat = output2[0].flatten().cpu()
        # mcc2 = matthews_corrcoef(label2_flat, output2_flat)
        # mcc_values2.append(mcc2)
        dice1 = f1_score(label1_flat, output1_flat)
        # dice_values1.append(dice1)
        dice2 = f1_score(label2_flat, output2_flat)
        # dice_values2.append(dice2)
        label3_flat = label1[1].flatten()
        output3_flat = output1[1].flatten().cpu()
        label4_flat = label2[1].flatten()
        output4_flat = output2[1].flatten().cpu()
        dice3 = f1_score(label3_flat, output3_flat)
        # dice_values1.append(dice3)
        dice4 = f1_score(label4_flat, output4_flat)
        # dice_values2.append(dice4)
        print(f"Average OC1 Dice: {dice1:.4f}")
        print(f"Average OD1 Dice: {dice2:.4f}")
        print(f"Average OC2 Dice: {dice3:.4f}")
        print(f"Average OD2 Dice: {dice4:.4f}")

        # # 绘制ROC曲线的准备
        # # 计算FPR, TPR 和 阈值
        # output1 = output[:, :1, :, :]
        # output2 = output[:, 1:2, :, :]
        # output2 = 1 - output2
        # label1_flat = label1.flatten()
        # output1_flat = output1.flatten().cpu()
        # label2_flat = label2.flatten()
        # output2_flat = output2.flatten().cpu()
        # #加入表中
        # all_labels1.append(label1_flat.flatten())
        # all_outputs1.append(output1_flat.flatten())
        # all_labels2.append(label2_flat.flatten())
        # all_outputs2.append(output2_flat.flatten())



        # fe1 = fe
        # for i in range(fe1.shape[0]):
        #     # 将图像数据类型转换为 float32
        #     x_np = fe1[i].detach().cpu().numpy()
        #     x_np = x_np.astype(np.float32)
        #     # 分离通道
        #     channels = cv2.split(x_np)
        #     # 对每个通道执行 DCT 变换
        #     dct_channels = [cv2.dct(channel) for channel in channels]
        #     # 合并变换后的通道
        #     dct_image = cv2.merge(dct_channels)
        #     fe1[i] = torch.tensor(dct_image, requires_grad=True).cuda()
        # output[output > threshold] = 1
        # output[output < threshold] = 0
        # output1 = output[:, :1, :, :]
        # output2 = output[:, 1:2, :, :]
        # output2[output2 == 0] = -1
        # output2[output2 == 1] = 0
        # output2[output2 == -1] = 1
        # img1 = img[0][:, :1, :, :]
        # img2 = img[0][:, 1:2, :, :]
        # dice1 = 1 - dice_loss(output1, img1, avg=False, epsilon=dice_epsilon)
        # binaryzation_dice_list1 = binaryzation_dice_list1 + dice1.tolist()
        # dice2 = 1 - dice_loss(output2, img2, avg=False, epsilon=dice_epsilon)
        # binaryzation_dice_list2 = binaryzation_dice_list2 + dice2.tolist()

        '''# 处理中间过程结果
        edge[edge > threshold] = 1
        edge[edge < threshold] = 0
        edge1 = edge[:, :1, :, :]
        edge2 = edge[:, 1:2, :, :]
        edge2[edge2 == 0] = -1
        edge2[edge2 == 1] = 0
        edge2[edge2 == -1] = 1
        edge = torch.cat([edge1, edge2], dim=1).float()

        log[log > threshold] = 1
        log[log < threshold] = 0
        log1 = log[:, :1, :, :]
        log2 = log[:, 1:2, :, :]
        log2[log2 == 0] = -1
        log2[log2 == 1] = 0
        log2[log2 == -1] = 1
        log = torch.cat([log1, log2], dim=1).float()'''

        # pred1 = torch.tensor(output1, dtype=torch.int, device="cpu")
        # pred1 = torch.reshape(pred1, (pred1.shape[0], 1, pred1.shape[-2] * pred1.shape[-1]))
        # gt_s1 = torch.tensor(img1, dtype=torch.int, device="cpu")
        # gt_s1 = torch.reshape(gt_s1, (gt_s1.shape[0], 1, gt_s1.shape[-2] * gt_s1.shape[-1]))
        # pred2 = torch.tensor(output2, dtype=torch.int, device="cpu")
        # pred2 = torch.reshape(pred2, (pred2.shape[0], 1, pred2.shape[-2] * pred2.shape[-1]))
        # gt_s2 = torch.tensor(img2, dtype=torch.int, device="cpu")
        # gt_s2 = torch.reshape(gt_s2, (gt_s2.shape[0], 1, gt_s2.shape[-2] * gt_s2.shape[-1]))
        # for i in range(len(output1)):

            # precision1 = precision(gt_s1[i][0], pred1[i][0])
            # recall1 = recall_score(gt_s1[i][0], pred1[i][0], zero_division=1.0)
            # accuracy1 = accuracy(gt_s1[i][0], pred1[i][0])
            # f1_1 = f1_score(gt_s1[i][0], pred1[i][0], zero_division=1.0)
            # jaccard1 = jaccard_score(gt_s1[i][0], pred1[i][0], zero_division=1.0)
            #print("precision1", precision1)
            #print("recall1", recall1)
            #print("accuracy1", accuracy1)
        #     print("f1_1", f1_1)
        #     #print("jaccard1", jaccard1)
        # for i in range(len(output2)):
        #     precision2 = precision(gt_s2[i][0], pred2[i][0])
        #     recall2 = recall_score(gt_s2[i][0], pred2[i][0], zero_division=1.0)
        #     accuracy2 = accuracy(gt_s2[i][0], pred2[i][0])
        #     f1_2 = f1_score(gt_s2[i][0], pred2[i][0], zero_division=1.0)
        #     jaccard2 = jaccard_score(gt_s2[i][0], pred2[i][0], zero_division=1.0)
            #print("precision2", precision2)
            #print("recall2", recall2)
            #print("accuracy2", accuracy2)
            # print("f1_2", f1_2)
            #print("jaccard2", jaccard2)

        '''绿色是绘图部分'''
        for i in range(img[0].shape[0]):
            if key_press == 113:  # 113 means key=="q"
                break
            gt_tensor1 = img[0][i][0].cpu().detach().numpy()
            gt_tensor2 = img[0][i][1].cpu().detach().numpy()
            temp2 = img[1][i].cpu().detach().numpy()
            temp31 = output1[i][0].cpu().detach().numpy()
            temp32 =output2[i][0].cpu().detach().numpy()
            # ted1 = edge[i][0].cpu().detach().numpy()
            # ted2 = edge[i][1].cpu().detach().numpy()
            # tlog1 = log[i][0].cpu().detach().numpy()
            # tlog2 = log[i][1].cpu().detach().numpy()

            # tfe = fe[i].cpu().detach().numpy()
            # tfe = fe.detach().cpu().squeeze().numpy().mean(axis=0)
            # plt.figure(cnt + 20)
            # plt.title("in_feature" + str(cnt))
            # plt.imshow(tfe)
            # plt.savefig(os.path.join(img_save_path, "in_feature" + str(cnt) + ".png"))
            # plt.savefig(os.path.join(img_save_path, "in_feature" + str(cnt) + ".eps"))
            #
            # tfe1 = fe1.detach().cpu().squeeze().numpy().mean(axis=0)
            # plt.figure(cnt + 20)
            # plt.title("in_feature1" + str(cnt))
            # plt.imshow(tfe1,cmap='rainbow')
            # plt.savefig(os.path.join(img_save_path, "in_feature1_" + str(cnt) + ".png"))
            # plt.savefig(os.path.join(img_save_path, "in_feature1_" + str(cnt) + ".eps"))



            # tfe2 = fe2.detach().cpu().squeeze().numpy().mean(axis=0)
            # plt.figure(cnt + 20)
            # plt.title("in_feature2_" + str(cnt))
            # plt.imshow(tfe2)
            # plt.savefig(os.path.join(img_save_path, "in_feature2_" + str(cnt) + ".png"))
            # plt.savefig(os.path.join(img_save_path, "in_feature2_" + str(cnt) + ".eps"))

            gt1 = np.expand_dims(gt_tensor1, axis=-1)
            s = np.zeros(gt1.shape, dtype=np.float)
            gt1 = np.concatenate([gt1, s, s], axis=-1)
            gt2 = np.expand_dims(gt_tensor2, axis=-1)
            s = np.zeros(gt2.shape, dtype=np.float)
            gt2 = np.concatenate([s, gt2, s], axis=-1)
            gt22 = np.expand_dims(gt_tensor2, axis=-1)
            gt22 = np.concatenate([gt22, s, s], axis=-1)

            predict1 = np.expand_dims(temp31, axis=-1)
            predict1 = np.concatenate([predict1, s, s], axis=-1)
            predict11 = np.expand_dims(temp31, axis=-1)
            predict11 = np.concatenate([s, predict11, s], axis=-1)
            predict2 = np.expand_dims(temp32, axis=-1)
            predict2 = np.concatenate([s, predict2, s], axis=-1)

            # ped1 = np.expand_dims(ted1, axis=-1)
            # ped1 = np.concatenate([ped1, s, s], axis=-1)
            # ped11 = np.expand_dims(ted1, axis=-1)
            # ped11 = np.concatenate([s, ped11, s], axis=-1)
            # ped2 = np.expand_dims(ted2, axis=-1)
            # ped2 = np.concatenate([s, ped2, s], axis=-1)
            #
            # plog1 = np.expand_dims(tlog1, axis=-1)
            # plog1 = np.concatenate([plog1, s, s], axis=-1)
            # plog11 = np.expand_dims(tlog1, axis=-1)
            # plog11 = np.concatenate([s, plog11, s], axis=-1)
            # plog2 = np.expand_dims(tlog2, axis=-1)
            # plog2 = np.concatenate([s, plog2, s], axis=-1)
            #
            # plog3 = np.expand_dims(tlog1, axis=-1)
            # plog3 = np.concatenate([s, plog3, s], axis=-1)
            # plog33 = np.expand_dims(tlog1, axis=-1)
            # plog33 = np.concatenate([s, plog33, s], axis=-1)
            # plog4 = np.expand_dims(tlog2, axis=-1)
            # plog4 = np.concatenate([s, s, plog4], axis=-1)

            # pfe = np.expand_dims(tfe, axis=-1)
            # s1 = np.zeros(pfe.shape, dtype=np.float)
            # pfe = np.concatenate([s1, pfe, s1], axis=-1)

            if np.sum(predict1) == 0 and np.sum(gt_tensor1) == 0:
                continue
            gt = cv2.addWeighted(gt1, 1, gt2, 1, 0)
            predict = cv2.addWeighted(predict1, 1, predict2, 1, 0)
            # plog = cv2.addWeighted(plog1, 1, plog2, 1, 0)
            # pplog = cv2.addWeighted(plog3, 1, plog4, 1, 0)
            # ped = cv2.addWeighted(ped1, 1, ped2, 1, 0)
            # mix1 = cv2.addWeighted(gt1, 1, predict11, 1, 0)
            # mix2 = cv2.addWeighted(gt22, 1, predict2, 1, 0)
            # plt.imshow(predict)
            # plt.show()
            # plt.imshow(mix1)
            # plt.show()
            # plt.imshow(mix2)
            # plt.show()
            # cv2.imshow("raw", temp2)
            # cv2.imshow("GT", gt)
            # cv2.imshow("predict", predict)
            # key_press = cv2.waitKey()

            if os.path.exists(img_save_path) is False:
                os.mkdir(img_save_path)

            # imsave(os.path.join(img_save_path, str(cnt) + "tfe.eps"), tfe)
            # imsave(os.path.join(img_save_path, str(cnt) + "tfe.png"), tfe)
            # imsave(os.path.join(img_save_path, str(cnt) + "plog.eps"), plog)
            # imsave(os.path.join(img_save_path, str(cnt) + "plog.png"), plog)
            # imsave(os.path.join(img_save_path, str(cnt) + "pplog.eps"), pplog)
            # imsave(os.path.join(img_save_path, str(cnt) + "pplog.png"), pplog)
            # imsave(os.path.join(img_save_path, str(cnt) + "ped.eps"), ped)
            # imsave(os.path.join(img_save_path, str(cnt) + "ped.png"), ped)
            # imsave(os.path.join(img_save_path, str(cnt) + "mix1.eps"), mix1)
            # imsave(os.path.join(img_save_path, str(cnt) + "mix1.png"), mix1)
            # imsave(os.path.join(img_save_path, str(cnt) + "mix2.eps"), mix2)
            # imsave(os.path.join(img_save_path, str(cnt) + "mix2.png"), mix2)
            # imsave(os.path.join(img_save_path, str(cnt) + "raw.eps"), temp2)
            # imsave(os.path.join(img_save_path, str(cnt) + "raw.jpg"), temp2)
            imsave(os.path.join(img_save_path, str(cnt) + "GT.eps"), gt)
            imsave(os.path.join(img_save_path, str(cnt) + "GT.png"), gt)
            imsave(os.path.join(img_save_path, str(cnt) + "predict.eps"), predict)
            imsave(os.path.join(img_save_path, str(cnt) + "predict.png"), predict)

            cnt += 1

        # print("un-binaryzation_uncertain", sum(unbinaryzation_uncertain_list) / len(unbinaryzation_uncertain_list))
        # print("un-binaryzation_dice_avg", sum(unbinaryzation_dice_list) / len(unbinaryzation_dice_list))

    #合并

    # all_labels1 = np.concatenate(all_labels1)
    # all_outputs1 = np.concatenate(all_outputs1)
    # all_labels2 = np.concatenate(all_labels2)
    # all_outputs2 = np.concatenate(all_outputs2)
    #
    # pack1 = [[all_outputs1[i], all_labels1[i]] for i in range(len(all_outputs1))]
    # pack1.sort()
    # pack_np1 = np.array(pack1)
    # all_outputs1 = pack_np1[:, 0]
    # all_labels1 = pack_np1[:, 1]
    #
    # pack2 = [[all_outputs2[i], all_labels2[i]] for i in range(len(all_outputs2))]
    # pack2.sort()
    # pack_np2 = np.array(pack2)
    # all_outputs2 = pack_np2[:, 0]
    # all_labels2 = pack_np2[:, 1]
    #
    # # 计算数组的长度
    # length = len(all_labels1)
    #
    # # 提取后二十分之一的数据
    # all_outputs1 = all_outputs1[-(length // 30):]
    # all_labels1 = all_labels1[-(length // 30):]
    # all_outputs2 = all_outputs2[-(length // 25):]
    # all_labels2 = all_labels2[-(length // 25):]
    #
    #
    # fpr1, tpr1, thresholds1 = roc_curve(all_labels1, all_outputs1)
    # fpr2, tpr2, thresholds2 = roc_curve(all_labels2, all_outputs2)
    # # 计算AUC (Area Under Curve)
    # roc_auc1 = auc(fpr1, tpr1)
    # roc_auc2 = auc(fpr2, tpr2)
    # # 绘制ROC曲线1
    # plt.figure()
    # plt.plot(fpr1, tpr1, color='blue', lw=2, label='ROC curve (area = %0.4f)' % roc_auc1)
    # plt.plot([0, 1], [0, 1], color='darkorange', lw=2, linestyle='--')
    # # 调整坐标轴限制以放大曲线细节
    # plt.xlim([0.0, 1.0])  # 只显示假阳性率（FPR）在0到0.2之间的部分
    # plt.ylim([0.0, 1.0])  # 只显示真阳性率（TPR）在0.8到1.0之间的部分
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel('FPR', fontsize = 14)
    # plt.ylabel('TPR', fontsize = 14)
    # plt.title('ROC of OC on the REFUGE2 dataset', fontsize = 16)
    # plt.legend(loc="lower right", fontsize = 14)
    #
    #
    # # 添加均匀的网格线，每0.2一个网格
    # plt.grid(which='both', linestyle='--', linewidth=0.5)
    # plt.xticks(np.arange(0, 1.1, 0.2))
    # plt.yticks(np.arange(0, 1.1, 0.2))
    #
    # # 保存图像为EPS和PNG格式
    # plt.savefig('roc_curve_oc_re.eps', format='eps')
    # plt.savefig('roc_curve_oc_re.png', format='png')
    # plt.show()
    #
    # # 绘制ROC曲线2
    # plt.figure()
    # plt.plot(fpr2, tpr2, color='blue', lw=2, label='ROC curve (area = %0.4f)' % roc_auc2)
    # plt.plot([0, 1], [0, 1], color='darkorange', lw=2, linestyle='--')
    # # 调整坐标轴限制以放大曲线细节
    # plt.xlim([0.0, 1.0])  # 只显示假阳性率（FPR）在0到0.2之间的部分
    # plt.ylim([0.0, 1.0])  # 只显示真阳性率（TPR）在0.8到1.0之间的部分
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.xlabel('FPR', fontsize=14)
    # plt.ylabel('TPR', fontsize=14)
    # plt.title('ROC of OD on the REFUGE2 dataset', fontsize=16)
    # plt.legend(loc="lower right", fontsize=14)
    #
    # plt.grid(which='both', linestyle='--', linewidth=0.5)
    # plt.xticks(np.arange(0, 1.1, 0.2))
    # plt.yticks(np.arange(0, 1.1, 0.2))
    #
    # # 保存图像为EPS和PNG格式
    # plt.savefig('roc_curve_od_re.eps', format='eps')
    # plt.savefig('roc_curve_od_re.png', format='png')
    # plt.show()
    # average_mcc1 = np.mean(mcc_values1)
    # print(f"Average OC MCC: {average_mcc1:.4f}")
    # average_mcc2 = np.mean(mcc_values2)
    # print(f"Average OD MCC: {average_mcc2:.4f}")
    # average_dice1 = np.mean(dice_values1)
    # print(f"Average OC dice:", average_dice1)
    # average_dice2 = np.mean(dice_values2)
    # print(f"Average OD dice:", average_dice2)