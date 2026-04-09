import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Fusion_dataset
from FusionNet import FusionNet
import os

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Fusion_dataset("train")
    print("Dataset size:",len(dataset))

    loader = DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0)

    model = FusionNet(output=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    criterion = nn.MSELoss()

    epochs = 2

    for epoch in range(epochs):

        print("Starting Epoch",epoch+1)

        for i,(vis,ir,name) in enumerate(loader):

            vis = vis.to(device)
            ir = ir.to(device)

            optimizer.zero_grad()

            # RGB → Y
            Y = 0.299*vis[:,0:1,:,:] + 0.587*vis[:,1:2,:,:] + 0.114*vis[:,2:3,:,:]

            fused = model(Y,ir)

            loss = criterion(fused,Y)

            loss.backward()

            optimizer.step()

            print("Batch",i,"Loss",loss.item())

        print("Epoch finished")

    os.makedirs("model",exist_ok=True)

    torch.save(model.state_dict(),"model/fusion_model.pth")

    print("Training Finished")

if __name__=="__main__":
    train()