import os

vis_dir = "datasets/LLVIP/visible/test"
ir_dir = "datasets/LLVIP/infrared/test"

vis_files = set(os.listdir(vis_dir))
ir_files = set(os.listdir(ir_dir))

common = vis_files.intersection(ir_files)

print("Visible images:", len(vis_files))
print("Infrared images:", len(ir_files))
print("Matching pairs:", len(common))

# Optional cleaning
for f in vis_files - common:
    os.remove(os.path.join(vis_dir, f))

for f in ir_files - common:
    os.remove(os.path.join(ir_dir, f))

print("Dataset cleaned. Final pairs:", len(common))