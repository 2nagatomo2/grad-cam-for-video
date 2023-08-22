from dataloader import make_datapath_list, VideoTransform, get_label_id_dictionary, VideoDataset
from model import ECO_Lite

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

# グラフのスタイルを指定
plt.style.use('seaborn-darkgrid')

batch_size = 4
weight_decay = 0.05
learning_rate = 0.0001
num_epochs = 10

#video_listの作成
train_root_path = './revised-data/left-handed/train/'
train_video_list = make_datapath_list(train_root_path)

val_root_path = './revised-data/left-handed/val/'
val_video_list = make_datapath_list(val_root_path)

#前処理の設定
resize, crop_size = 224, 224
mean, std = [104, 117, 123], [1, 1, 1]
video_transform = VideoTransform(resize, crop_size, mean, std)

#ラベル辞書の作成
label_dictionary_path = './revised-data/pitching.csv'
label_id_dict, id_label_dict = get_label_id_dictionary(label_dictionary_path)

#Datasetの作成（画像）
train_dataset = VideoDataset(train_video_list, label_id_dict, num_segments=16,
                             phase="train", transform=video_transform,
                             img_tmpl='image_{:05d}.jpg')

val_dataset = VideoDataset(val_video_list, label_id_dict, num_segments=16,
                             phase="val", transform=video_transform,
                             img_tmpl='image_{:05d}.jpg')

#DataLoaderにする
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle = True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle = False)

#Networkの準備，lass関数，最適化手法
device = torch.device("cuda")
net = ECO_Lite()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

#パラメータの読み込み
#net.load_state_dict(torch.load('model_weight.pth'))

def train_epoch(model, optimizer, criterion, dataloader, device):
    train_loss = 0
    train_acc = 0
    model.train()
    for i, (imgs_transformeds, labels, label_ids, dir_path) in enumerate(train_dataloader):
        images, labels = imgs_transformeds.to(device), label_ids.to(device)
        optimizer.zero_grad()
        outputs, out2 = model(images)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += torch.sum(preds == labels)
    train_loss = train_loss / len(train_dataset)
    train_acc = float(train_acc) / len(train_dataset)
    return train_loss, train_acc, out2

def inference(model, optimizer, criterion, dataloader, device):
    model.eval()
    test_loss=0
    test_acc=0

    with torch.no_grad():
        for i, (imgs_transformeds, labels, label_ids, dir_path) in enumerate(val_dataloader):
            images, labels = imgs_transformeds.to(device), label_ids.to(device)
            outputs, out2 = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item()
            test_acc += torch.sum(preds == labels)
        test_loss = test_loss / len(val_dataset)
        test_acc = float(test_acc) / len(val_dataset)
    return test_loss, test_acc, out2

def run(num_epochs, optimizer, criterion, device):
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(num_epochs):
        train_loss, train_acc, out2 = train_epoch(net, optimizer, criterion, train_dataloader, device)
        test_loss, test_acc, out2 = inference(net, optimizer, criterion, val_dataloader, device)

        print(f'Epoch [{epoch+1}], train_Loss : {train_loss:.4f},train_Acc : {train_acc:.4f}, test_Loss : {test_loss:.4f}, test_Acc : {test_acc:.4f}')
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list, out2

train_loss_list, train_acc_list, test_loss_list, test_acc_list, out2 = run(num_epochs, optimizer, criterion, device)
"""
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
print(output_weights.size(), out2.size(), cam.size())

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

# 結果を表示
print(cam.size())
"""
print(min(train_loss_list))
print(min(test_loss_list))
print(max(train_acc_list))
print(max(test_acc_list))

# パラメータの保存
#torch.save(net.state_dict(), 'model_weight.pth')

plt.figure(figsize=(6,6))      #グラフ描画用

#以下グラフ描画
plt.plot(range(num_epochs), train_loss_list)
plt.plot(range(num_epochs), test_loss_list, c='#00ff00')
plt.xlim(0, num_epochs)
plt.ylim(0, 1.0)
plt.xlabel('num_epochs')
plt.ylabel('LOSS')
plt.legend(['train loss', 'val loss'])
plt.title('loss')
plt.savefig("loss.jpg")
plt.clf()
plt.show()

plt.plot(range(num_epochs), train_acc_list)
plt.plot(range(num_epochs), test_acc_list, c='#00ff00')
plt.xlim(0, num_epochs)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("acc.jpg")
plt.show()
