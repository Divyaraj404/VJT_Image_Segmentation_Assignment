import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A block that performs two consecutive convolutions with ReLU activation.
    Optionally, it can include Batch Normalization after each convolution.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),  # Optional: Helps with training stability
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Optional
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    A down-sampling block that applies max pooling followed by a double convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    An up-sampling block that upsamples and then concatenates the corresponding encoder feature map.
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Here we use ConvTranspose2d for learned upsampling
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1: upsampled input from the decoder
        # x2: corresponding feature map from the encoder (skip connection)
        x1 = self.up(x1)
        # Input sizes might not match exactly due to rounding; we crop x2 if needed.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match dimensions of x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=11, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder: initial convolution block and subsequent down-sampling blocks
        self.inc = DoubleConv(n_channels, 64)      # initial block
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # For the final down, check for bilinear flag; if bilinear,
        # reduce channel count differently to preserve computation
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder: up-sampling and feature concatenation blocks
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final 1x1 convolution to produce the segmentation map
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # size: [B, 64, H, W]
        x2 = self.down1(x1)   # size: [B, 128, H/2, W/2]
        x3 = self.down2(x2)   # size: [B, 256, H/4, W/4]
        x4 = self.down3(x3)   # size: [B, 512, H/8, W/8]
        x5 = self.down4(x4)   # size: [B, 1024, H/16, W/16] or [B, 512, ...] if bilinear
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)  # Merge x5 and x4
        x = self.up2(x, x3)   # Merge with x3
        x = self.up3(x, x2)   # Merge with x2
        x = self.up4(x, x1)   # Merge with x1
        
        # Output segmentation map, shape: [B, n_classes, H, W]
        logits = self.outc(x)
        return logits

# For testing the UNet architecture, you can add the following in a main guard:
if __name__ == "__main__":
    # Create a dummy input tensor with batch size 1 and image size 256x256
    dummy_input = torch.randn(1, 3, 256, 256)
    model = UNet(n_channels=3, n_classes=11)
    output = model(dummy_input)
    print("UNet output shape:", output.shape)  # Expected shape: [1, 11, 256, 256]
