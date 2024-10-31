import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import gc

# Enhanced Feature Extraction Module (EFEM)
class EnhancedFeatureExtractionModule(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedFeatureExtractionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(in_channels * 3)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat([x1, x3, x5], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_out(x)
        return x

# Context-Aware Feature Aggregation (CAFA)
class ContextAwareFeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=21):
        super(ContextAwareFeatureAggregation, self).__init__()
        padding = kernel_size // 2
        self.large_kernel_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.residual_conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        large_kernel_out = self.large_kernel_conv(x)
        global_feat = self.global_avg_pool(x)
        global_feat = self.global_conv(global_feat)
        global_feat = F.interpolate(global_feat, size=large_kernel_out.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([large_kernel_out, global_feat], dim=1)
        x = self.bn(x)
        return self.relu(x + residual)

# Adaptive Fusion Module (AFM)
class AdaptiveFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveFusionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.fusion_weights = nn.Parameter(torch.ones(3, requires_grad=True))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = self.fusion_weights[0] * x1 + self.fusion_weights[1] * x3 + self.fusion_weights[2] * x5
        x = self.bn(x)
        x = self.relu(x)
        return x

# Depthwise Separable Convolution with Skip Connection
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip_connection(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x + identity

# Double Convolution Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        return self.conv(x)

# MSCAMNet Model for Skin Lesion Segmentation with Deep Supervision, EFEM, CAFA, and AFM
class MSCAMNet(nn.Module):
    def __init__(self, n_channels, n_classes=1):  
        super(MSCAMNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        
        self.encoder = smp.encoders.get_encoder(
            "timm-efficientnet-b2",
            in_channels=n_channels,
            depth=5,
            weights="imagenet"
        )

        
        self.efem1 = EnhancedFeatureExtractionModule(16)
        self.efem2 = EnhancedFeatureExtractionModule(24)
        self.efem3 = EnhancedFeatureExtractionModule(48)
        self.efem4 = EnhancedFeatureExtractionModule(120)
        self.efem5 = EnhancedFeatureExtractionModule(352)

        
        self.bottleneck = ContextAwareFeatureAggregation(352, 352)
        
        # Additional refinement blocks in the bottleneck
        self.refine1 = DoubleConv(704, 352)
        self.refine2 = DoubleConv(352, 352)

        # Decoder with dynamic upsampling and Adaptive Fusion Module
        self.up1 = nn.ConvTranspose2d(352, 120, kernel_size=2, stride=2)
        self.dec1 = AdaptiveFusionModule(472, 120)
        
        self.up2 = nn.ConvTranspose2d(120, 88, kernel_size=2, stride=2)
        self.dec2 = AdaptiveFusionModule(208, 88)
        
        self.up3 = nn.ConvTranspose2d(88, 48, kernel_size=2, stride=2)
        self.dec3 = AdaptiveFusionModule(96, 48)
        
        self.up4 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec4 = AdaptiveFusionModule(48, 24)
        
        
        self.out_conv = nn.Conv2d(24, n_classes, kernel_size=1)

        # Auxiliary outputs for deep supervision 
        self.aux_out1 = nn.Conv2d(352, n_classes, kernel_size=1)
        self.aux_out2 = nn.Conv2d(120, n_classes, kernel_size=1)
        self.aux_out3 = nn.Conv2d(48, n_classes, kernel_size=1)
    
    def forward(self, x):
        
        features = self.encoder(x)
        
        x0, x1, x2, x3, x4, x5 = features
        
        
        x2 = self.efem2(x2)
        x3 = self.efem3(x3)
        x4 = self.efem4(x4)
        x5 = self.efem5(x5)
        
        
        x6 = self.bottleneck(x5)
        
        
        x6 = self.refine1(x6)
        x6 = self.refine2(x6)
        
        
        x = self.up1(x6)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x5], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x4 = F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x4], dim=1)
        x = self.dec2(x)
        
        x = self.up3(x)
        x3 = F.interpolate(x3, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        x = self.up4(x)
        x2 = F.interpolate(x2, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.dec4(x)
        
        
        x = self.out_conv(x)
        
        # Auxiliary outputs for deep supervision
        aux1 = self.aux_out1(F.interpolate(x5, x.size()[2:], mode='bilinear', align_corners=True))
        aux2 = self.aux_out2(F.interpolate(x4, x.size()[2:], mode='bilinear', align_corners=True))
        aux3 = self.aux_out3(F.interpolate(x3, x.size()[2:], mode='bilinear', align_corners=True))
        
        return x, aux1, aux2, aux3


def get_model():
    return MSCAMNet(n_channels=3, n_classes=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    print(model)

    x = torch.randn((1, 3, 256, 256)).to(device)  

    try:
        logits, aux1, aux2, aux3 = model(x)
        print("Logits shape:", logits.shape)  
        print("Aux1 shape:", aux1.shape)
        print("Aux2 shape:", aux2.shape)
        print("Aux3 shape:", aux3.shape)
    except RuntimeError as e:
        print("RuntimeError:", e)

    batch_size = 1  
    data = torch.randn((batch_size, 3, 256, 256)).to(device)
    target = torch.randint(0, 2, (batch_size, 256, 256)).to(device)  

    try:
        outputs, aux1, aux2, aux3 = model(data)
        print("Model output shape:", outputs.shape)

        # Upsample model outputs to match input image size
        upsampled_logits = F.interpolate(outputs, size=target.shape[-2:], mode="bilinear", align_corners=False)
        print("Upsampled logits shape:", upsampled_logits.shape)
    except RuntimeError as e:
        print("RuntimeError during forward pass:", e)

    
    gc.collect()
    torch.cuda.empty_cache()
