from ultralytics import YOLO

def main():
    
    model = YOLO("yolo11x.pt")
    model.to("cuda")
    print("YOLO is using:", model.device)

    model.train(
        data=r"suitcase-30-12-2024.v2i.yolov11\data.yaml",
        epochs=50,
        batch=4,
        imgsz=640,
        cache=True,
        name="yolo11x_finetuned_suitcase",
        project="runs/train",
        pretrained=True
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    main()
