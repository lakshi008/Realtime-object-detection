import os
import shutil

source = r"C:\Users\Lakshitha P\runs\detect\sharp_object_detector2\weights\best.pt"
destination_dir = r"model"
destination = os.path.join(destination_dir, "weapon_detector.pt")

os.makedirs(destination_dir, exist_ok=True)
shutil.copy(source, destination)

print("✅ Model copied and renamed to:", destination)
