# model.py
import torch
from facenet_pytorch import InceptionResnetV1

# Khởi tạo model giống hệt trong file train
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Nếu có GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = resnet.to(device)
