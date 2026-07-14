#!/usr/bin/env python3
"""workload_video_analytics.py - Video Analytics Pipeline (CPU decode + GPU detection)
실행: python3 workload_video_analytics.py -t 120 --device cuda:0
목적: 실제 영상 분석 파이프라인을 시뮬레이션하여 CPU 비디오 디코딩 +
      전처리가 GPU detection throughput에 미치는 영향을 측정.

기존 workload_yolo.py와의 차이:
- yolo: ultralytics의 predict(stream=True)로 내부 최적화된 파이프라인 사용
- video_analytics: OpenCV로 직접 프레임 디코딩 + 멀티워커 CPU 전처리 후 YOLO inference
  → CPU 코어 수에 따라 전처리 병렬성이 변화 → throughput 차이 발생

CPU 작업: cv2.VideoCapture 디코딩 + resize + GaussianBlur + color normalization (멀티워커)
GPU 작업: YOLO detection inference
"""

import cv2
import numpy as np
import time
import argparse
from concurrent.futures import ThreadPoolExecutor


def preprocess_frame(frame, target_size=640):
    """실제 YOLO 전처리 파이프라인. OpenCV 연산은 GIL을 해제하므로 threading 가능."""
    # 1. Resize to model input size
    frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # 2. GaussianBlur (denoising — 실제 영상 분석에서 일반적)
    frame = cv2.GaussianBlur(frame, (5, 5), 1.0)

    # 3. BGR → RGB 변환 + float32 정규화
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0

    return frame


def read_frames_batch(cap, batch_size):
    """비디오에서 batch_size만큼 프레임 읽기. 끝나면 되감기."""
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser(description='Video analytics pipeline (CPU decode + GPU detection)')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                        help='Duration in seconds (default: 120)')
    parser.add_argument('--source', type=str, default='test_video.mp4',
                        help='Video source (default: test_video.mp4)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model file (default: yolov8n.pt)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='CUDA device (default: cuda:0)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Preprocessing worker threads (default: 4)')
    parser.add_argument('--frame-batch', type=int, default=8,
                        help='Frames to preprocess in parallel (default: 8)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='(unused, for compatibility)')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed")
        return

    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Duration: {args.timeout}s")
    print(f"Preprocessing workers: {args.workers}")
    print(f"Frame batch size: {args.frame_batch}")
    print(f"Pipeline: OpenCV decode → parallel CPU preprocess ({args.workers} threads) → YOLO GPU inference")

    # 비디오 열기
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.source}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_video_frames} frames @ {video_fps:.1f} fps")

    # YOLO 모델 로드
    print("Loading YOLO model...")
    model = YOLO(args.model)

    # Warmup
    print("Warming up...")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    model.predict(dummy, device=args.device, verbose=False)

    print("Starting video analytics...\n")

    start_time = time.time()
    total_frames = 0

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while time.time() - start_time < args.timeout:
                # 1. 프레임 배치 읽기 (sequential — video decoding)
                frames = read_frames_batch(cap, args.frame_batch)
                if not frames:
                    break

                # 2. 병렬 CPU 전처리 (ThreadPoolExecutor)
                # OpenCV 연산은 GIL을 해제하므로 실제 병렬 실행됨
                processed = list(pool.map(preprocess_frame, frames))

                # 3. GPU inference (sequential — GPU는 1개)
                for p in processed:
                    if time.time() - start_time >= args.timeout:
                        break
                    model.predict(
                        p,
                        device=args.device,
                        verbose=False,
                        save=False,
                    )
                    total_frames += 1

                if total_frames % 100 == 0 and total_frames > 0:
                    elapsed = time.time() - start_time
                    fps = total_frames / elapsed
                    print(f"  [{elapsed:.1f}s] frames: {total_frames} ({fps:.1f} fps)")

    except KeyboardInterrupt:
        pass

    cap.release()

    elapsed = time.time() - start_time
    print(f"\nDone. {total_frames} frames in {elapsed:.1f}s "
          f"({total_frames/elapsed:.1f} fps)")


if __name__ == '__main__':
    main()
