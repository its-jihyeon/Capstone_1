# ✅ 목표: G Loss < 0.1 & PSNR/SSIM 향상 | CBAM + Perceptual Loss + 안정화 구조

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import torchvision.models as models

# ===================== CBAM (Attention 모듈) =====================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.shared(x))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg, max], dim=1)
        return x * self.sigmoid(self.conv(concat))

class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# ===================== Encoder (U-Net + CBAM) =====================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                CBAM(out_c),
                nn.Conv2d(out_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(6, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, cover, secret):
        x = torch.cat((cover, secret), dim=1)
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        up3 = self.up3(b)
        d3 = self.dec3(torch.cat((up3, e3), dim=1))
        up2 = self.up2(d3)
        d2 = self.dec2(torch.cat((up2, e2), dim=1))
        up1 = self.up1(d2)
        d1 = self.dec1(torch.cat((up1, e1), dim=1))
        return self.tanh(self.final(d1))

# ===================== Discriminator =====================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)

# ===================== Perceptual Loss =====================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features[:16].eval()
        for param in vgg.parameters(): param.requires_grad = False
        self.vgg = vgg.half().to('cuda')  # T4 최적화: perceptual loss 속도 개선
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()

    def forward(self, x, y):
        x = (x + 1) / 2; y = (y + 1) / 2
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return nn.functional.mse_loss(self.vgg(x), self.vgg(y))

# ===================== Dataset =====================
class StegoDataset(Dataset):
    def __init__(self, cover_dir, secret_dir, transform):
        self.cover_paths = sorted([os.path.join(cover_dir, f) for f in os.listdir(cover_dir)])
        self.secret_paths = sorted([os.path.join(secret_dir, f) for f in os.listdir(secret_dir)])
        self.length = min(len(self.cover_paths), len(self.secret_paths))
        self.cover_paths = self.cover_paths[:self.length]
        self.secret_paths = self.secret_paths[:self.length]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        cover = Image.open(self.cover_paths[idx]).convert("RGB")
        secret = Image.open(self.secret_paths[idx]).convert("RGB")
        return self.transform(cover), self.transform(secret)

# ===================== 학습 =====================
cover_dir = "/content/drive/MyDrive/train/clean"
secret_dir = "/content/drive/MyDrive/train/stego"
save_dir = "/content/drive/MyDrive/steganography/trained_model"
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

batch_size = 4
epochs = 70
start_epoch = 1

# 데이터 로더
dataset = StegoDataset(cover_dir, secret_dir, transform)
subset_size = 8000  # 논문용 학습 데이터 수
subset_indices = list(range(subset_size))
dataset = torch.utils.data.Subset(dataset, subset_indices)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# 장비 설정 및 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder().to(device)
discriminator = Discriminator().to(device)
perceptual_loss = PerceptualLoss().to(device)

# 옵티마이저 및 손실 함수
optimizer_G = torch.optim.AdamW(encoder.parameters(), lr=3e-5, betas=(0.5, 0.999))
optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=3e-5, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()
scaler_G = GradScaler()
scaler_D = GradScaler()

# 학습 루프
for epoch in range(start_epoch, start_epoch + epochs):
    encoder.train(); discriminator.train()
    for cover, secret in tqdm(dataloader):
        cover, secret = cover.to(device, non_blocking=True), secret.to(device, non_blocking=True)
        valid = torch.ones((cover.size(0), 1), device=device)
        fake = torch.zeros((cover.size(0), 1), device=device)

        # Discriminator 학습
        optimizer_D.zero_grad()
        with autocast(device_type=device.type):
            stego = encoder(cover, secret).detach()
            d_real = discriminator(cover)
            d_fake = discriminator(stego)
            d_loss = (criterion(d_real, valid) + criterion(d_fake, fake)) / 2
        scaler_D.scale(d_loss).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()

        # Generator 학습
        optimizer_G.zero_grad()
        with autocast(device_type=device.type):
            stego = encoder(cover, secret)
            g_adv = criterion(discriminator(stego), valid)
            g_recon = mse_loss(stego, cover)
            g_percep = perceptual_loss(stego.half(), cover.half())  # AMP 성능 개선: perceptual 입력도 half 적용
            g_loss = 0.1 * g_adv + 0.7 * g_recon + 0.3 * g_percep
        scaler_G.scale(g_loss).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] ✅ G_loss: {g_loss.item():.4f} | D_loss: {d_loss.item():.4f}")

    # 10 에폭마다 저장
    if epoch % 10 == 0 or epoch == start_epoch + epochs - 1:
        encoder.eval()
        with torch.no_grad():
            c, s = next(iter(dataloader))
            out = encoder(c.to(device), s.to(device))[0].unsqueeze(0).cpu() * 0.5 + 0.5
            save_image(out, f"{save_dir}/stego_epoch{epoch}.png")
        torch.save(encoder.state_dict(), f"{save_dir}/generator_epoch{epoch}.pt")
        torch.save(discriminator.state_dict(), f"{save_dir}/discriminator_epoch{epoch}.pt")

print("🎯 학습 완료! 최종 모델 저장됨.")