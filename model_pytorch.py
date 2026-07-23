"""
PyTorch版本的手势识别CNN模型 - 增强版
包含残差连接、通道注意力机制和更深的网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块 (Squeeze-and-Excitation)"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # 合并注意力
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class ResidualBlock(nn.Module):
    """残差块 - 带BatchNorm和注意力机制"""
    
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 通道注意力
        self.use_attention = use_attention
        if use_attention:
            self.attention = ChannelAttention(out_channels)
        
        # 残差连接的shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.attention(out)
        
        out += identity
        out = F.relu(out)
        
        return out


class GestureCNN(nn.Module):
    """增强版手势识别CNN模型 - 包含残差连接和注意力机制"""
    
    def __init__(self, num_classes=10):
        super(GestureCNN, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        
        # 残差块组1 - 64通道
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1, use_attention=True),
            ResidualBlock(64, 64, stride=1, use_attention=False)
        )
        
        # 残差块组2 - 128通道
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_attention=True),  # 32x32 -> 16x16
            ResidualBlock(128, 128, stride=1, use_attention=True),
            ResidualBlock(128, 128, stride=1, use_attention=False)
        )
        
        # 残差块组3 - 256通道
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_attention=True),  # 16x16 -> 8x8
            ResidualBlock(256, 256, stride=1, use_attention=True),
            ResidualBlock(256, 256, stride=1, use_attention=False)
        )
        
        # 残差块组4 - 512通道
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2, use_attention=True),  # 8x8 -> 4x4
            ResidualBlock(512, 512, stride=1, use_attention=True)
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(512, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 初始特征提取
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        
        # 残差块组
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.dropout1(F.relu(self.bn_fc(self.fc1(x))))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def create_model(num_classes=10, pretrained_path=None):
    """
    创建模型
    
    Args:
        num_classes: 类别数量
        pretrained_path: 预训练模型路径
        
    Returns:
        model: PyTorch模型
    """
    model = GestureCNN(num_classes=num_classes)
    
    if pretrained_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(pretrained_path))
    elif pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
    
    return model


if __name__ == "__main__":
    # 测试模型
    model = GestureCNN(num_classes=10)
    print("模型结构:")
    print(model)
    print()
    
    # 测试前向传播
    dummy_input = torch.randn(1, 3, 64, 64)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
