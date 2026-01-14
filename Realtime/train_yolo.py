from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt


dataset_path = r"D:\SEMESTER 5\AIML\guns-knives-yolo"

print("Files in dataset folder:", os.listdir(dataset_path))
print("\nChecking data.yaml content:\n")
with open(os.path.join(dataset_path, "data.yaml")) as f:
    print(f.read())


train_images = len(os.listdir(os.path.join(dataset_path, 'train/images')))
valid_images = len(os.listdir(os.path.join(dataset_path, 'valid/images')))
print(f"\nTrain Images: {train_images}")
print(f"Validation Images: {valid_images}")


sample_image_path = os.path.join(dataset_path, "train/images", "--------_------_jpg.rf.0c6fff42bd233e5fb0e2ef448b4e4db1.jpg")
if os.path.exists(sample_image_path):
    img = cv2.imread(sample_image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("Replace 'example.jpg' with a real image name from your dataset.")


model = YOLO("yolov8n.pt")  

results = model.train(
    data=os.path.join(dataset_path, "data.yaml"),
    epochs=5,      
    imgsz=640,      
    batch=8,        
    name="sharp_object_detector"
)

print("\nTraining complete! Model saved in 'runs/detect/sharp_object_detector/weights/best.pt'")


trained_model = YOLO("runs/detect/sharp_object_detector/weights/best.pt")


test_path = os.path.join(dataset_path, "valid/images")
trained_model.predict(source=test_path, save=True, conf=0.5)

print("\n Predictions saved in 'runs/detect/predict' folder.")

