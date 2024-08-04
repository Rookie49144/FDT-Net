import os
import torch


## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 32
cosineLR = True
n_channels = 3
n_labels = 2
epochs = 200
img_size = 512
print_frequency = 1
save_frequency = 100
vis_frequency = 100
early_stopping_patience = 100
learning_rate = 1e-4
batch_size = 2

pretrain = False


train_dataset = ''
val_dataset = ''
test_dataset = ''




