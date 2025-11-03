# models/c3d.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class C3D(nn.Module):
    """
    Lightweight C3D. Input shape: (B, 3, T, H, W)
    Output: logits for N classes
    """
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        def conv_block(in_c, out_c, kernel=(3,3,3), pool=(1,2,2)):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=kernel, padding=(1,1,1)),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=pool)
            )

        self.features = nn.Sequential(
            conv_block(3, 64, pool=(1,2,2)),    # T unchanged
            conv_block(64, 128, pool=(2,2,2)),
            conv_block(128, 256, pool=(2,2,2)),
            conv_block(256, 512, pool=(2,2,2)),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
