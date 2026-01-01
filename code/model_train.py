"""
Training script for fine-tuning YOLOv8n on cricket ball dataset.
This should NOT be run on test videos.
"""

def train_model():
    """
    Train YOLOv8n on cricket ball dataset from Roboflow.
    
    """
    from ultralytics import YOLO
    
    # Load pretrained YOLOv8n
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data='dataset/data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='cpu',  # Use 'cuda' if GPU available during training
        patience=10,
        save=True,
        project='runs/train',
        name='cricket_ball',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        weight_decay=0.0005,
        workers=2  # Low for CPU
    )
    
    print("Training complete!")
    print(f"Best model saved at: runs/train/cricket_ball/weights/best.pt")

if __name__ == '__main__':
    train_model()
