from ultralytics import YOLO

def main():
    # Cargar el modelo base preentrenado (YOLOv8 nano)
    model = YOLO("yolov8n.pt")

    # Iniciar el fine-tuning usando el dataset configurado
    # Importante: ajusta los "epochs" y el "batch_size" según la capacidad de tu GPU.
    results = model.train(
        data="dataset.yaml",
        epochs=50,       # Número de épocas, ajústalo según el resultado
        batch=16,        # Tamaño de lote, si te da OOM bájalo a 8 o 4
        imgsz=640,       # Tamaño de la imagen para entrenamiento
        project="runs",  # Carpeta donde se guardarán los resultados
        name="ambulance_detection", # Nombre de la ejecución
        device="cpu"     # Usando CPU ya que CUDA no está disponible
    )

    print("Entrenamiento completado. El mejor modelo se guardó en runs/ambulance_detection/weights/best.pt")

if __name__ == "__main__":
    main()
