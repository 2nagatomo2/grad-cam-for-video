import torch
import torch.nn.functional as F
from torch.autograd import Function
from model import ECO_Lite

# Grad-CAM用のクラスを定義
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None

    def _save_gradient(self, grad):
        self.gradient = grad

    def generate_cam(self, input, target_class):
        output = self.model(input)
        output_tensor = output[0]  # タプルから出力テンソルを取得
        output_index = output_tensor.argmax(dim=1)

        # 目的のクラスのスコアに対する勾配を計算
        self.model.zero_grad()
        one_hot = torch.zeros_like(output_tensor)
        one_hot[0][target_class] = 1
        gradients = torch.autograd.grad(output_tensor, self.target_layer, grad_outputs=one_hot, retain_graph=True)[0]
        
        # 勾配情報のチェックと保存
        if self.gradient is None:
            self.gradient = torch.zeros_like(input)
        
        self._save_gradient(gradients)

        # 勾配情報から重みを計算
        gradients = self.gradient.cpu().numpy()[0]
        activations = self.target_layer.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2, 3))
        cam = np.sum(weights[:, np.newaxis, np.newaxis, np.newaxis] * activations, axis=0)
        cam = np.maximum(cam, 0)  # ReLUを適用して負の値を0にする
        cam = cam / np.max(cam)  # 正規化

        return cam



# Grad-CAMを生成するための準備
model = ECO_Lite()
target_layer = model.eco_3d.res_3d_5.res5b_2  # Grad-CAMの対象となる層
grad_cam = GradCAM(model, target_layer)

# 入力データと目的のクラスを指定してGrad-CAMを生成
input = torch.randn(1, 16, 3, 224, 224)  # 入力データの例
target_class = 0  # 目的のクラスのインデックス
cam = grad_cam.generate_cam(input, target_class)
