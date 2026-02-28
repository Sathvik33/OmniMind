import cv2
import os
import uuid


class VideoService:

    def __init__(self, vision_service, collection_manager):
        self.vision_service = vision_service
        self.collection_manager = collection_manager

    def process(self, video_path: str, source_name: str, job_id: str, app):

        app.state.video_jobs[job_id] = "processing"

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            cap.release()
            app.state.video_jobs[job_id] = "failed"
            return

        frame_interval = int(fps * 2)

        documents = []
        ids = []
        metadatas = []

        frame_count = 0
        prev_frame = None
        prev_description = None
        segment_start = None
        segment_end = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:

                timestamp_sec = int(frame_count / fps)

                should_process = True

                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, frame)
                    mean_diff = diff.mean()

                    if mean_diff < 5:
                        should_process = False

                if should_process:
                    temp_image_path = f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(temp_image_path, frame)

                    description = self.vision_service.describe(temp_image_path)

                    os.remove(temp_image_path)

                    if prev_description is None:
                        segment_start = timestamp_sec
                        segment_end = timestamp_sec + 2
                        prev_description = description

                    else:
                        if description.strip() == prev_description.strip():
                            segment_end = timestamp_sec + 2
                        else:
                            documents.append(prev_description)
                            ids.append(str(uuid.uuid4()))
                            metadatas.append({
                                "source": source_name,
                                "modality": "video",
                                "start_time": segment_start,
                                "end_time": segment_end
                            })

                            segment_start = timestamp_sec
                            segment_end = timestamp_sec + 2
                            prev_description = description

                    prev_frame = frame

            frame_count += 1

        cap.release()

        if prev_description is not None:
            documents.append(prev_description)
            ids.append(str(uuid.uuid4()))
            metadatas.append({
                "source": source_name,
                "modality": "video",
                "start_time": segment_start,
                "end_time": segment_end
            })

        if documents:
            self.collection_manager.add_documents(documents, ids, metadatas)

        app.state.video_jobs[job_id] = "completed"