import os
import io
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Request, Depends, Form, Response, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import engine, Base, get_db
from models import DetectionHistory
import shutil
import openpyxl
import cv2  
import numpy as np  



app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Монтируем статику
os.makedirs("static/results", exist_ok=True)
os.makedirs("static/videos", exist_ok=True)
os.makedirs("reports", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# Загружаем модель
model = YOLO("yolov8s.pt")

# Инициализация базы данных
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        # Создаём таблицы, если их нет
        await conn.run_sync(Base.metadata.create_all)

def draw_boxes_on_image(image: Image.Image, boxes):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        conf = box["conf"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"Motorcycle {conf:.2f}", fill="red", font=font)
    return image

async def save_detection_history(db: AsyncSession, filename: str, detections: int, media_type: str):
    history = DetectionHistory(filename=filename, detections=detections, media_type=media_type)
    db.add(history)
    await db.commit()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: AsyncSession = Depends(get_db)):
    # Получаем последние 10 записей
    result = await db.execute(select(DetectionHistory).order_by(DetectionHistory.timestamp.desc()).limit(10))
    history = result.scalars().all()
    return templates.TemplateResponse("index.html", {"request": request, "history": history})


async def process_video_full(input_path: str, output_path: str, max_frames=100):
    """
    Обрабатывает видео кадр за кадром, накладывает рамки детекции и сохраняет новый видеофайл.
    max_frames - ограничение по количеству кадров для ускорения (можно убрать или увеличить).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для mp4

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Запускаем детекцию на кадре (YOLO принимает numpy array BGR)
        results = model(frame)

        # Рисуем рамки для мотоциклов (класс 3)
        for box in results[0].boxes:
            cls_id = int(box.cls.cpu().numpy())
            if cls_id == 3:
                xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                conf = box.conf.cpu().numpy()[0]
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Motorcycle {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()


@app.post("/detect", response_class=HTMLResponse)
async def detect(request: Request, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    # Проверяем тип файла
    content = await file.read()
    filename = file.filename.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detections_count = 0

    # Обработка изображений
    if filename.endswith((".jpg", ".jpeg", ".png")):
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            return templates.TemplateResponse("index.html", {"request": request, "error": "Невозможно открыть изображение"})
        
        results = model(image)

        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.cpu().numpy())
                if cls_id == 3:  # Мотоцикл
                    xyxy = box.xyxy.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    boxes.append({"xyxy": xyxy, "conf": conf})

        detections_count = len(boxes)
        image = draw_boxes_on_image(image, boxes)
        output_filename = f"static/results/result_{timestamp}.jpg"
        image.save(output_filename)

        await save_detection_history(db, output_filename, detections_count, "image")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result_image": "/" + output_filename.replace("\\", "/"),
            "detections": detections_count
        })

    # Обработка видео
    elif filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        # Сохраняем видео
        video_path = f"static/videos/{timestamp}_{filename}"
        with open(video_path, "wb") as f:
            f.write(content)

        try:
            # Обрабатываем видео с помощью модели (выделим первые 30 кадров для примера)
            # ultralytics YOLO поддерживает видео, но для простоты обработаем только 1-й кадр
            results = model(video_path)

            # Получаем первый кадр с результатами
            frame = results[0].orig_img
            boxes = []
            for box in results[0].boxes:
                cls_id = int(box.cls.cpu().numpy())
                if cls_id == 3:
                    xyxy = box.xyxy.cpu().numpy()[0]
                    conf = box.conf.cpu().numpy()[0]
                    boxes.append({"xyxy": xyxy, "conf": conf})

            detections_count = len(boxes)

            # Нарисуем рамки на кадре
            frame_pil = Image.fromarray(frame)
            frame_pil = draw_boxes_on_image(frame_pil, boxes)

            output_filename = f"static/results/result_{timestamp}.jpg"
            frame_pil.save(output_filename)
        
            output_video_path = f"static/results/result_video_{timestamp}.mp4"
            await process_video_full(video_path, output_video_path, max_frames=100)

            await save_detection_history(db, video_path, detections_count, "video")

            return templates.TemplateResponse("index.html", {
                "request": request,
                "result_image": "/" + output_filename.replace("\\", "/"),
                "detections": detections_count,
                "video_uploaded": "/" + video_path.replace("\\", "/"),
                "video_processed": "/" + output_video_path.replace("\\", "/")  
            })
        except Exception as e:
            return templates.TemplateResponse("index.html", {"request": request, "error": f"Ошибка при обработке видео: {str(e)}"})

    else:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Неподдерживаемый формат файла"})

@app.get("/report/excel", response_class=FileResponse)
async def generate_report(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(DetectionHistory).order_by(DetectionHistory.timestamp.desc()))
    history = result.scalars().all()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "История детекций"

    headers = ["ID", "Имя файла", "Количество мотоциклов", "Тип медиа", "Дата и время"]
    ws.append(headers)

    for record in history:
        ws.append([
            record.id,
            record.filename,
            record.detections,
            record.media_type,
            record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        ])

    report_path = f"reports/detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    wb.save(report_path)

    return FileResponse(report_path, filename="detection_report.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

