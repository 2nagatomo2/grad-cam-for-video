from dataloader import make_datapath_list, VideoTransform, get_label_id_dictionary, VideoDataset
from model import ECO_Lite
from train-left2 import train_epoch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

#Networkの準備，lass関数，最適化手法
device = torch.device("cuda")
net = ECO_Lite()
net = net.to(device)
criterion = nn.CrossEntropyLoss()

#CAM
output_weights = None
for module in net.modules():
    if isinstance(module, nn.Linear):
        output_weights = module.weight.clone().detach()
        break


# 特徴マップを重みで重み付けして和を計算
out2 = out2.squeeze(0)  # バッチ次元を削除し、形状を(512, 4, 7, 7)に変更
out2 = out2.to(output_weights.device)

cam = torch.zeros(2, 4, 7, 7)  # CAMの初期化
cam = cam.to(out2.device)
for j in range(output_weights.size(0)):  # クラスの数（2）に対してループ
    for c in range(cam.size(1)):  # チャンネルの数（4）に対してループ
        for h in range(cam.size(2)):  # 高さ次元（7）に対してループ
            for w in range(cam.size(3)):  # 幅次元（7）に対してループ
                cam[j, c, h, w] += torch.sum(output_weights[j] * out2[:, c, h, w])

# CAMの正規化
cam = torch.softmax(cam, dim=0)

# ヒートマップを作成し、カラーマップを適用して表示
for j in range(cam.shape[0]):  # クラスのループ
    # クラスごとのヒートマップを取得
    class_cam = cam[j].unsqueeze(0)  # サイズ: [1, 4, 7, 7]

    # クラスごとに最大値を求める
    class_cam_max = torch.max(class_cam.view(1, -1), dim=1)[0]  # サイズ: [1]

    # ヒートマップを作成
    heatmap = class_cam_max.view(1, 1, 1, 1)  # サイズ: [1, 1, 1, 1]

    # ヒートマップを入力サイズに拡大
    heatmap = F.interpolate(heatmap, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze().cpu().numpy()

    # カラーマップを適用してヒートマップを表示
    plt.imshow(heatmap, cmap='jet')
    plt.axis('off')
    plt.show()
