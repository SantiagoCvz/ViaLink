# SYSCOM — Sistema de Detección Vehicular con YOLO + SAHI

Sistema de detección en tiempo real de ambulancias y vehículos desde cámara IP RTSP.

## Arquitectura

```
Cámara IP (RTSP)
    │
    ▼
FFmpeg subprocess  ←── Conversión RTSP → raw BGR frames
    │
    ▼
YOLO v8 + SAHI  ←── Inferencia con sliced detection
    │
    ▼
Flask /stream  ←── MJPEG multipart stream (HTTP)
    │
    ▼
Navegador web  ←── <img src="/stream"> + polling /status
```

## Instalación rápida

### 1. Requisitos del sistema
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y ffmpeg python3-pip

# macOS
brew install ffmpeg
```

### 2. Instalar dependencias Python
```bash
pip install -r requirements.txt
```

### 3. Iniciar el servidor
```bash
bash start.sh
# o directamente:
python3 server.py
```

### 4. Abrir en el navegador
```
http://localhost:5000
```

---

## Configuración RTSP

En `server.py`, línea 20:
```python
RTSP_URL = "rtsp://admin:Syscom2026@169.254.18.91:554/ISAPI/Streaming/channels/1"
```

Asegúrate de que la red tenga acceso a `169.254.18.91` (link-local).
En Linux puedes asignar una IP en ese rango:
```bash
sudo ip addr add 169.254.18.1/16 dev eth0
```

---

## Cómo funciona la conversión RTSP → HTTP

1. **FFmpeg subprocess** (`RTSPReader`): invoca FFmpeg con `-rtsp_transport tcp`
   y salida `-f rawvideo -pix_fmt bgr24 pipe:1`. Lee frames crudos byte a byte.
2. **OpenCV fallback** (`OpenCVReader`): si FFmpeg no está disponible, usa
   `cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)` con buffer size reducido.
3. **Flask `/stream`**: genera un `multipart/x-mixed-replace` MJPEG stream.
   El navegador lo consume con una simple etiqueta `<img>`.

---

## Detección de Ambulancias

YOLOv8 (COCO) no tiene clase "ambulance". Se usa un **heurístico de color HSV**:
- Si un vehículo detectado (car/bus/truck) tiene >15% pixeles rojos → ambulance
- Si tiene >55% pixeles blancos → ambulance
- Con SAHI activo, se aplica detección por slices para objetos pequeños

Para mayor precisión, entrena YOLOv8 con un dataset propio de ambulancias:
```bash
yolo detect train data=ambulances.yaml model=yolov8n.pt epochs=50
```

---

## Endpoints API

| Endpoint    | Descripción                                      |
|-------------|--------------------------------------------------|
| `GET /`     | Interfaz web                                     |
| `GET /stream` | Stream MJPEG (multipart/x-mixed-replace)      |
| `GET /status` | JSON con estado, FPS, conteos de detecciones  |

### Ejemplo `/status`
```json
{
  "stream_status": "live",
  "fps": 14.8,
  "frame_count": 1240,
  "total_detections": 37,
  "detection_count": { "car": 28, "ambulance": 2, "truck": 5, "bus": 2 },
  "model": "SAHI+YOLOv8n",
  "uptime": 83
}
```

---

## Ajuste de parámetros

| Variable             | Archivo    | Descripción                                 |
|----------------------|------------|---------------------------------------------|
| `RTSP_URL`           | server.py  | URL del stream RTSP                         |
| `CONF_THRESHOLD`     | server.py  | Umbral de confianza (default 0.35)          |
| `DETECTION_INTERVAL` | server.py  | Detección cada N frames (default 3)         |
| `TARGET_FPS`         | server.py  | FPS objetivo del stream (default 15)        |
| `FRAME_WIDTH/HEIGHT` | server.py  | Resolución de captura (default 1280×720)    |
