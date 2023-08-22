import torch
import torch.nn as nn
from eco import ECO_2D, ECO_3D

class ECO_Lite(nn.Module):
    def __init__(self):
        super(ECO_Lite, self).__init__()
        # 2D Netモジュール
        self.eco_2d = ECO_2D()

        # 3D Netモジュール
        self.eco_3d = ECO_3D()
        # クラス分類の全結合層
        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)
    
    def forward(self, x):
        '''
        入力xはtorch.Size([batch_num, num_segments=16, 3, 224, 224])
        '''

        # 入力xの各次元のサイズを取得する
        bs, ns, c, h, w = x.shape

        # xを(bs*ns, c, h, w)にサイズ変換する
        out = x.view(-1, c, h, w)
        # （注釈）
        # PyTorchのConv2Dは入力のサイズが(batch_num, c, h, w)しか受け付けないため
        # (batch_num, num_segments, c, h, w)は処理できない
        # 今は2次元画像を独立に処理するので、num_segmentsはbatch_numの次元に押し込んでも良いため
        # (batch_num×num_segments, c, h, w)にサイズを変換する

        # 2D Netモジュール 出力torch.Size([batch_num×16, 96, 28, 28])
        out = self.eco_2d(out)

        # 2次元画像をテンソルを3次元用に変換する
        # num_segmentsをbatch_numの次元に押し込んだものを元に戻す
        out = out.view(-1, ns, 96, 28, 28)

        # 3D Netモジュール 出力torch.Size([batch_num, 512])
        out, out2 = self.eco_3d(out)
        # クラス分類の全結合層　出力torch.Size([batch_num, class_num=2])
        out = self.fc_final(out)
        #out2 = nn.
        return out, out2

