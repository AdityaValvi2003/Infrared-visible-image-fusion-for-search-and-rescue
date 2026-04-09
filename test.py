import torch
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from FusionNet import FusionNet
from PIL import Image
import os

import torch
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from FusionNet import FusionNet
from PIL import Image
import os

base_dir = "/content/drive/MyDrive/BTech_Project/Data/LLVIP"

print("Visible test:", len(os.listdir(os.path.join(base_dir,"visible","test"))))
print("Infrared test:", len(os.listdir(os.path.join(base_dir,"infrared","test"))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionNet(output=1).to(device)
model.load_state_dict(torch.load("model/fusion_model.pth",map_location=device))
model.eval()

dataset = Fusion_dataset("test")
print("Dataset size:",len(dataset))

loader = DataLoader(dataset,batch_size=1)

os.makedirs("results",exist_ok=True)

for i,(vis,ir,name) in enumerate(loader):

    print("Processing:",name[0])

    vis = vis.to(device)
    ir = ir.to(device)

    Y = 0.299*vis[:,0:1,:,:] + 0.587*vis[:,1:2,:,:] + 0.114*vis[:,2:3,:,:]

    fused = model(Y,ir)

    img = fused.squeeze().cpu().detach().numpy()
    img = (img*255).astype("uint8")

    Image.fromarray(img).save("results/"+name[0])

print("Fusion complete")

base_dir = "/content/drive/MyDrive/BTech_Project/Data/LLVIP"

print("Visible test:", len(os.listdir(os.path.join(base_dir,"visible","test"))))
print("Infrared test:", len(os.listdir(os.path.join(base_dir,"infrared","test"))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FusionNet(output=1).to(device)
model.load_state_dict(torch.load("model/fusion_model.pth",map_location=device))
model.eval()

dataset = Fusion_dataset("test")
print("Dataset size:",len(dataset))

loader = DataLoader(dataset,batch_size=1)

os.makedirs("results",exist_ok=True)

for i,(vis,ir,name) in enumerate(loader):

    print("Processing:",name[0])

    vis = vis.to(device)
    ir = ir.to(device)

    Y = 0.299*vis[:,0:1,:,:] + 0.587*vis[:,1:2,:,:] + 0.114*vis[:,2:3,:,:]

    fused = model(Y,ir)

    img = fused.squeeze().cpu().detach().numpy()
    img = (img*255).astype("uint8")

    Image.fromarray(img).save("results/"+name[0])

print("Fusion complete")