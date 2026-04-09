import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from natsort import natsorted

class Fusion_dataset(Dataset):

    def __init__(self, split):

        base_dir = "/content/drive/MyDrive/BTech_Project/Data/LLVIP"

        if split == "train":
            self.vis_dir = os.path.join(base_dir, "visible", "train")
            self.ir_dir = os.path.join(base_dir, "infrared", "train")
        else:
            self.vis_dir = os.path.join(base_dir, "visible", "test")
            self.ir_dir = os.path.join(base_dir, "infrared", "test")

        vis_files = set(os.listdir(self.vis_dir))
        ir_files = set(os.listdir(self.ir_dir))

        # keep only matching filenames
        self.files = natsorted(list(vis_files.intersection(ir_files)))

        print("Total paired images:", len(self.files))

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        name = self.files[index]

        vis_path = os.path.join(self.vis_dir, name)
        ir_path = os.path.join(self.ir_dir, name)

        vis = Image.open(vis_path).convert("RGB")
        ir = Image.open(ir_path).convert("L")

        vis = self.transform(vis)
        ir = self.transform(ir)

        return vis, ir, name