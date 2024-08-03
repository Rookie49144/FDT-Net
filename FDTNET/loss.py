import torch
import torch.nn as nn
import copy
import re


def dice_loss(predict, ground_truth, avg=True, epsilon=1e-0):  # type tensor shape Batch*Channel*DataShape
    shape = [i for i in range(1, len(predict.shape))]
    temp1 = torch.mul(predict, ground_truth)
    temp1 = torch.sum(temp1, shape)
    temp1 = temp1 * 2 + epsilon
    temp2 = torch.sum(predict, shape)
    temp3 = torch.sum(ground_truth, shape)
    temp4 = temp2 + temp3 + epsilon
    temp5 = torch.div(temp1, temp4)
    if avg:
        return torch.mean(1 - temp5)
    else:
        return 1 - temp5


def entropy_loss(inputs, avg=True, clamp_eps=1e-6):
    inputs = torch.clamp(inputs, min=clamp_eps, max=1 - clamp_eps)
    loss = inputs + torch.log(inputs)
    if avg:
        loss = torch.mean(loss)
    else:
        shape = [i for i in range(1, len(inputs.shape))]
        loss = torch.sum(loss, dim=shape) / torch.cumprod(inputs.shape[1:], dim=0)
    return loss


def unbalance_boundary_loss(predict, dist_map, avg=True):
    loss = predict * dist_map
    if avg:
        return torch.sum(loss) / predict.shape[-1]
    else:
        shape = [i for i in range(1, len(predict.shape))]
        return torch.sum(loss, shape) / predict.shape[-1]


class ProposeBoundaryLoss(nn.Module):
    def __init__(self, interval_max=4, avg=True):
        super().__init__()
        self.interval_max = interval_max
        self.avg = avg

    def forward(self, prediction, ground_truth, dist_map, epsilon=1e-1):
        temp = ground_truth.view(len(ground_truth), -1)
        with torch.no_grad():
            batch_mask, _ = torch.max(temp, dim=1)
            # batch_mask = torch.where(temp1 >= 0.1, 1, 0)
            mask = self.boundary_mask(dist_map)
        dice_map = torch.mul(prediction, ground_truth)
        temp1 = mask * dice_map
        temp2 = (prediction + ground_truth) * mask
        shape = [i for i in range(1, len(ground_truth.shape))]
        temp1 = torch.sum(temp1, shape)
        temp2 = torch.sum(temp2, shape)
        temp3 = (temp1 * 2 + epsilon) / (temp2 + epsilon)
        temp3 = batch_mask * temp3
        if self.avg:
            return torch.mean(1 - temp3)
        else:
            return 1 - temp3

    def boundary_mask(self, x):
        # S(x)*(1-S(x)) > 1e-3 : max|x|=5
        x = self.interval_max / 5 * x
        return torch.sigmoid(x) * (1 - torch.sigmoid(x)) * 4


class RegressLoss(nn.Module):
    def __init__(self, avg=True, negative_mask=True, negative_mask_rate=1e-2, scale=1):
        super().__init__()
        self.avg = avg
        self.negative_mask = negative_mask
        self.negative_mask_rate = negative_mask_rate
        self.scale = scale
        self.kernel = torch.tensor([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]],
                                   dtype=torch.float32).view(1, 1, 3, 3)

        self.mse_loss = nn.MSELoss(reduce=False)

    def forward(self, predict, ground_truth):
        if re.search("cuda", str(self.kernel.device)) is None:
            if re.search("cuda", str(ground_truth.device)) is not None:
                self.kernel = self.kernel.cuda()

        with torch.no_grad():
            reg_gt = nn.functional.conv2d(ground_truth, self.kernel, padding=1)
            reg_gt = reg_gt / self.scale

        loss = self.mse_loss(predict, reg_gt)

        if self.negative_mask:
            mask = torch.where(reg_gt > 0, 1, self.negative_mask_rate)
            loss = loss * mask

        if self.avg is False:
            shape = [i for i in range(1, len(ground_truth.shape))]
            # loss = torch.mean(loss, shape)
            loss = torch.sum(loss, dim=shape) / ground_truth[-1]
        else:
            # loss = torch.mean(loss)
            loss = torch.sum(loss) / ground_truth.shape[-1]

        return loss


class RegressGaussianLoss(nn.Module):
    def __init__(self, avg=True, negative_mask=True, negative_mask_rate=1e-2, scale=0.3):
        super().__init__()
        self.avg = avg
        self.negative_mask = negative_mask
        self.negative_mask_rate = negative_mask_rate
        self.scale = scale
        self.kernel = torch.ones([7, 7], dtype=torch.float32).view(1, 1, 7, 7)

        self.mse_loss = nn.MSELoss(reduce=False)

    def forward(self, predict, ground_truth):
        if re.search("cuda", str(self.kernel.device)) is None:
            if re.search("cuda", str(ground_truth.device)) is not None:
                self.kernel = self.kernel.cuda()

        with torch.no_grad():
            reg_gt = nn.functional.conv2d(ground_truth, self.kernel, padding=1)
            reg_gt = reg_gt * self.scale

        loss = self.mse_loss(predict, reg_gt)

        if self.negative_mask:
            mask = torch.where(reg_gt > 0, 1, self.negative_mask_rate)
            loss = loss * mask

        if self.avg is False:
            shape = [i for i in range(1, len(ground_truth.shape))]
            loss = torch.mean(loss, shape)
        else:
            loss = torch.mean(loss)

        return loss

    def gauss_gener(self, sigma, kernel_size=5):
        mean = torch.tensor([(kernel_size - 1) / 2, (kernel_size - 1) / 2], dtype=torch.float32)
        kernel = torch.ones([kernel_size, kernel_size], dtype=torch.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = torch.exp(-((i - mean[0]) ** 2 + (j - mean[1]) ** 2) / 2 / sigma)
        kernel = kernel / torch.sum(kernel)
        self.kernel = kernel.view(1, 1, kernel_size, kernel_size)


class PromptLoss(nn.Module):
    def __init__(self, avg=True, need_activate=True):
        super().__init__()
        self.avg = avg
        self.need_activate = need_activate
        self.avg_pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bce = nn.BCELoss(reduction="none")

    def forward(self, prompt_list, ground_truth):
        if self.avg:
            bce_loss_ = torch.zeros([1], dtype=torch.float32)
            dice_loss_ = torch.zeros([1], dtype=torch.float32)
        else:
            bce_loss_ = torch.zeros([len(ground_truth)], dtype=torch.float32)
            dice_loss_ = torch.zeros([len(ground_truth)], dtype=torch.float32)

        if re.search("cuda", str(ground_truth.device)) is not None:
            bce_loss_ = bce_loss_.cuda()
            dice_loss_ = dice_loss_.cuda()

        for prompt in prompt_list:
            bce, dice = self.img_forward(prompt, ground_truth)
            bce_loss_ = bce_loss_ + bce
            dice_loss_ = dice_loss_ + dice
            with torch.no_grad():
                ground_truth = self.avg_pooling(ground_truth)
        return bce_loss_, dice_loss_

    def img_forward(self, predict, ground_truth):
        if self.need_activate is True:
            predict = torch.sigmoid(predict)
        bce_loss = self.bce(predict, ground_truth)
        if self.avg is True:
            bce_loss = torch.mean(bce_loss)
        else:
            shape = [i for i in range(1, len(predict.shape))]
            bce_loss = torch.mean(bce_loss, shape)
        loss = [bce_loss, dice_loss(predict, ground_truth, self.avg)]
        return loss


if __name__ == "__main__":
    x = torch.tensor([0.5, 0.5])
    y=entropy_loss(x)
    print(y)
