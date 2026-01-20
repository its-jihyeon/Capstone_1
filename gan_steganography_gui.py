import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QFrame
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

# ===================== TextEncoder =====================
class TextEncoder(nn.Module):
    def __init__(self, text_length=16):
        super().__init__()
        self.embedding = nn.Embedding(128, 256)
        self.fc = nn.Sequential(
            nn.Linear(text_length * 256, 512 * 512),
            nn.Tanh()
        )

    def forward(self, text):
        x = self.embedding(text)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 1, 512, 512)
        x = x.repeat(1, 3, 1, 1)
        return x

# ===================== CBAM Block =====================
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.ca = self.ChannelAttention(channels, reduction)
        self.sa = self.SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction):
            super().__init__()
            self.shared = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(channels // reduction, channels, 1, bias=False)
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return x * self.sigmoid(self.shared(x))

    class SpatialAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x_cat = torch.cat([avg_out, max_out], dim=1)
            return x * torch.sigmoid(self.conv(x_cat))

# ===================== Encoder =====================
def conv_cbam_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, 1, 1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        CBAMBlock(out_c),
        nn.Conv2d(out_c, out_c, 3, 1, 1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_cbam_block(6, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_cbam_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_cbam_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = conv_cbam_block(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_cbam_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_cbam_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_cbam_block(128, 64)
        self.final = nn.Conv2d(64, 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, cover, secret):
        x = torch.cat((cover, secret), dim=1)
        e1 = self.enc1(x); p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        b = self.bottleneck(p3)
        up3 = self.up3(b); d3 = self.dec3(torch.cat((up3, e3), dim=1))
        up2 = self.up2(d3); d2 = self.dec2(torch.cat((up2, e2), dim=1))
        up1 = self.up1(d2); d1 = self.dec1(torch.cat((up1, e1), dim=1))
        return self.tanh(self.final(d1))

# ===================== GUI =====================
class StegoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GAN Steganography")
        self.setFixedSize(580, 600)
        self.setStyleSheet("background-color: #f2f2f2;")
        self.image_path = None
        
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.generator_path = os.path.join(self.base_dir, "generator_epoch.pt")
        self.result_dir = os.path.join(self.base_dir, "result")
        self.setup_ui()

    def setup_ui(self):
        outer_layout = QVBoxLayout()
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 3px solid #3399ff;
                border-radius: 12px;
            }
        """)
        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        frame.setLayout(layout)

        title = QLabel("GAN Steganography")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("border: none;")
        layout.addWidget(title)

        img_row = QHBoxLayout()
        img_label = QLabel("이미지 선택")
        img_label.setFont(QFont("Arial", 11))
        img_label.setStyleSheet("border: none;")

        self.img_input = QLineEdit()
        self.img_input.setPlaceholderText("이미지를 선택하세요.")
        self.img_input.setFixedHeight(32)
        self.img_input.setFont(QFont("Arial", 11))
        self.img_input.setStyleSheet("background-color: #e6e6e6; border: none;")

        img_btn = QPushButton("찾기")
        img_btn.setFixedHeight(40)
        img_btn.setFixedWidth(80)
        img_btn.setFont(QFont("Arial", 10, QFont.Bold))
        img_btn.setStyleSheet("""
            QPushButton {
                background-color: #3399ff;
                color: white;
                border-radius: 6px;
                font-size: 12pt;
            }
        """)
        img_btn.clicked.connect(self.select_image)

        img_row.addWidget(img_label)
        img_row.addWidget(self.img_input)
        img_row.addWidget(img_btn)
        layout.addLayout(img_row)

        self.img_preview = QLabel()
        self.img_preview.setAlignment(Qt.AlignCenter)
        self.img_preview.setFixedSize(256, 256)
        self.img_preview.setStyleSheet("border: none; margin: 0px; padding: 0px;")
        self.img_preview.hide()
        layout.addWidget(self.img_preview, alignment=Qt.AlignCenter)

        layout.addSpacing(6)

        text_label = QLabel("숨길 텍스트 입력 (16자 이내 영어 텍스트, 공백 포함)")
        text_label.setFont(QFont("Arial", 11))
        text_label.setFixedHeight(20)
        text_label.setStyleSheet("border: none; margin: 0px; padding: 0px;")
        layout.addWidget(text_label, alignment=Qt.AlignLeft)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("텍스트를 입력하세요.")
        self.text_input.setFont(QFont("Arial", 11))
        self.text_input.setFixedHeight(32)
        self.text_input.setFixedWidth(500)
        self.text_input.setStyleSheet("background-color: #e6e6e6; border: none;")
        layout.addWidget(self.text_input, alignment=Qt.AlignLeft)

        layout.addSpacing(3)

        self.generate_btn = QPushButton("Stego Image 생성")
        self.generate_btn.setFixedHeight(42)
        self.generate_btn.setFixedWidth(280)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #3399ff;
                color: white;
                font-weight: bold;
                font-size: 12pt;
                border-radius: 6px;
            }
        """)
        self.generate_btn.clicked.connect(self.generate_stego)
        layout.addWidget(self.generate_btn, alignment=Qt.AlignCenter)

        self.result_labels = QHBoxLayout()
        self.clean_text = QLabel("Clean Image")
        self.clean_text.setFont(QFont("Arial", 10))
        self.clean_text.setAlignment(Qt.AlignCenter)
        self.clean_text.hide()

        self.stego_text = QLabel("Stego Image")
        self.stego_text.setFont(QFont("Arial", 10))
        self.stego_text.setAlignment(Qt.AlignCenter)
        self.stego_text.hide()

        self.result_labels.addWidget(self.clean_text)
        self.result_labels.addWidget(self.stego_text)
        layout.addLayout(self.result_labels)

        self.result_images = QHBoxLayout()
        self.clean_img = QLabel()
        self.clean_img.setFixedSize(256, 256)
        self.clean_img.setAlignment(Qt.AlignCenter)
        self.clean_img.setStyleSheet("border: none;")
        self.clean_img.hide()

        self.stego_img = QLabel()
        self.stego_img.setFixedSize(256, 256)
        self.stego_img.setAlignment(Qt.AlignCenter)
        self.stego_img.setStyleSheet("border: none;")
        self.stego_img.hide()

        self.result_images.addWidget(self.clean_img)
        self.result_images.addWidget(self.stego_img)
        layout.addLayout(self.result_images)

        outer_layout.addWidget(frame)
        self.setLayout(outer_layout)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "PNG Images (*.png)")
        if file_path:
            self.image_path = file_path
            self.img_input.setText(file_path)

            self.clean_img.hide()
            self.stego_img.hide()
            self.clean_text.hide()
            self.stego_text.hide()
            self.text_input.clear()

            pixmap = QPixmap(file_path).scaled(256, 256, Qt.KeepAspectRatio)
            self.img_preview.setPixmap(pixmap)
            self.img_preview.setVisible(True)

    def generate_stego(self):
        if not self.image_path or not self.text_input.text().strip():
            QMessageBox.warning(self, "입력 오류", "이미지와 텍스트를 모두 입력하세요.")
            return

        text = self.text_input.text().strip()

        if any('\uac00' <= c <= '\ud7a3' for c in text):
            QMessageBox.critical(self, "입력 오류", "영어 텍스트만 가능합니다.")
            return

        if len(text) > 16:
            QMessageBox.critical(self, "입력 오류", "텍스트는 16자 이내로 입력하세요. (공백 포함)")
            return

        text = text.ljust(16)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = Encoder().to(device)
        generator.load_state_dict(torch.load(self.generator_path, map_location=device))
        generator.eval()
        text_encoder = TextEncoder().to(device).eval()

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        img = Image.open(self.image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        text_tensor = torch.tensor([[ord(c) for c in text]], dtype=torch.long).to(device)
        with torch.no_grad():
            secret = text_encoder(text_tensor)
            stego = generator(img_tensor, secret)

        stego_np = stego.squeeze().cpu().numpy()
        stego_np = (stego_np * 0.5 + 0.5).clip(0, 1).transpose(1, 2, 0)
        stego_img = Image.fromarray((stego_np * 255).astype(np.uint8))

        os.makedirs(self.result_dir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(self.result_dir, f"stego_{i}.png")):
            i += 1
        out_path = os.path.join(self.result_dir, f"stego_{i}.png")
        stego_img.save(out_path)

        self.img_preview.clear()
        self.img_preview.hide()

        clean_pixmap = QPixmap(self.image_path).scaled(256, 256, Qt.KeepAspectRatio)
        stego_pixmap = QPixmap(out_path).scaled(256, 256, Qt.KeepAspectRatio)

        self.clean_img.setPixmap(clean_pixmap)
        self.stego_img.setPixmap(stego_pixmap)

        self.clean_img.show()
        self.stego_img.show()
        self.clean_text.show()
        self.stego_text.show()

        print(f"\u2705 Stego 이미지 저장 완료: {out_path}")

# ===================== 실행 =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = StegoGUI()
    gui.show()
    sys.exit(app.exec_())



