import cv2
import os
import uuid
from pathlib import Path


class VideoService:

    def __init__(self, vision_service, collection_manager):
        self.vision_service = vision_service
        self.collection_manager = collection_manager

    def process(self, video_path: str, source_name: str):

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            cap.release()
            return

        frame_interval = int(fps * 2)

        documents = []
        ids = []
        metadatas = []

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp_sec = int(frame_count / fps)

                temp_image_path = f"temp_frame_{frame_count}.jpg"
                cv2.imwrite(temp_image_path, frame)

                description = self.vision_service.describe(temp_image_path)

                os.remove(temp_image_path)

                documents.append(description)
                ids.append(str(uuid.uuid4()))
                metadatas.append({
                    "source": source_name,
                    "modality": "video",
                    "start_time": timestamp_sec,
                    "end_time": timestamp_sec + 2
                })

            frame_count += 1

        cap.release()

        if documents:
            self.collection_manager.add_documents(documents, ids, metadatas)