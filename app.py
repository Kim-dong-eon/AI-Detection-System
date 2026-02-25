# ========================== 기본 모듈 ===========================
from flask import Flask, render_template, jsonify, request, Response, redirect, session, url_for, stream_with_context, send_from_directory
from pymongo import MongoClient
from functools import wraps
import os
from datetime import datetime, timedelta
import bcrypt
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화
import subprocess
import time
from ultralytics import YOLO
import shutil
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import cv2
import threading
import atexit
import gridfs
from bson.objectid import ObjectId
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from bson.objectid import ObjectId
from kakao_api import send_kakao_message  # 카카오 메시지 전송 함수 import

# OpenCV 디스플레이 관련 설정
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
cv2.setNumThreads(1)

def dummy(*args, **kwargs):
    pass

# OpenCV 디스플레이 관련 함수 비활성화
cv2.imshow = dummy
cv2.namedWindow = dummy
cv2.waitKey = dummy
cv2.destroyAllWindows = dummy
cv2.startWindowThread = dummy
cv2.destroyWindow = dummy
cv2.createWindow = dummy
cv2.resizeWindow = dummy
cv2.moveWindow = dummy
cv2.setWindowProperty = dummy
cv2.getWindowProperty = dummy
cv2.getWindowImageRect = dummy

# ========================== 로깅 설정 ===========================
# 로그 파일 디렉토리 생성
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로그 파일 경로 설정
log_file = os.path.join(log_dir, 'video_cleanup.log')

# 로거 설정
logger = logging.getLogger('video_cleanup')
logger.setLevel(logging.INFO)

# 파일 핸들러 설정
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 로그 포맷 설정
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# rich와 mmengine의 progress bar 비활성화
import sys, io
from contextlib import contextmanager
from mmengine.utils.progressbar import ProgressBar
from rich.progress import Progress

# progress bar 비활성화를 위한 함수
@contextmanager
def suppress_stdout():
    new_target = io.StringIO()
    old_target, sys.stdout = sys.stdout, new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target

# mmengine의 ProgressBar 클래스 재정의
class SilentProgressBar(ProgressBar):
    def start(self):
        pass
    
    def update(self, num_tasks=None):
        pass
        
    def finish(self):
        pass

# mmengine의 기본 progress bar 교체
ProgressBar = SilentProgressBar

# rich의 progress bar 비활성화
Progress.get_default_columns = lambda self: []

# ========================== AI 및 영상 관련 모듈 ===========================
import numpy as np
from collections import deque
import pygame
from threading import Thread
from mmaction.apis.inferencers.mmaction2_inferencer import MMAction2Inferencer

# ========================== Flask 앱 설정 ===========================
app = Flask(__name__, template_folder='templates')
app.secret_key = 'your-secret-key'

# ========================== MongoDB 연결 ===========================
try:
    # 전역 클라이언트는 video_surveillance에 연결 유지 (예: 이벤트 로그)
    client = MongoClient('mongodb://localhost:27017/')
    db = client['video_surveillance']  # 데이터베이스 이름 (이벤트 로그 등)
    events_collection = db['events']  # 이벤트 로그 컬렉션

    # 영상 데이터베이스 클라이언트 별도 생성
    video_client = MongoClient('mongodb://localhost:27017/')
    video_db = video_client['save_to_data_video'] # 영상 저장 데이터베이스 이름

    # 인덱스 생성 (video_db에 적용)
    video_db.fs.files.create_index([("upload_date", -1)])  # 시간 역순 정렬을 위한 인덱스 (GridFS)
    video_db.fs.files.create_index([("risk_level", 1)])  # 위험도 기준 검색을 위한 인덱스 (GridFS)
    events_collection.create_index([("timestamp", -1)])  # 이벤트 시간 역순 정렬 (video_surveillance)

    logger.info("MongoDB 연결 성공 (video_surveillance)")
    logger.info("MongoDB 연결 성공 (save_to_data_video)")
except Exception as e:
    logger.error(f"MongoDB 연결 실패: {e}")
    raise

# ========================== 모델 경로 설정 ===========================
MODEL_CONFIG = {
    "abnormal": {
        "config": "C:/py/model/Abnormal_Behavior/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb.py",
        "weights": "C:/py/model/Abnormal_Behavior/tsm_best_acc_epoch7.pth"
    },
    "purchase": {
        "config": "C:/py/model/Purchase_Behavior/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb.py",
        "weights": "C:/py/model/Purchase_Behavior/best_acc_top1_epoch_10.pth"
    }
}

# ========================== 모델 초기화 ===========================
model_abnormal = MMAction2Inferencer(
    rec=MODEL_CONFIG["abnormal"]["config"],
    rec_weights=MODEL_CONFIG["abnormal"]["weights"],
    device="cuda:0",
    input_format="array"
)
model_abnormal.visualizer = None

model_purchase = MMAction2Inferencer(
    rec=MODEL_CONFIG["purchase"]["config"],
    rec_weights=MODEL_CONFIG["purchase"]["weights"],
    device="cuda:0",
    input_format="array"
)
model_purchase.visualizer = None

# ========================== 상수 및 스트림 캐시 ===========================
SEQUENCE_LENGTH = 8  # TSM 모델에 적합한 프레임 수로 변경
HYPER_VALUE_STEAL = 0.97
HYPER_VALUE_WARNING = 0.90
ALERT_SOUND_PATH = "static/audio/alert_sound.mp3"
VIDEO_SAVE_PATH = "static/save_to_data"
CCTV_NAME = "CCTV-1"
NO_PERSON_FRAMES_THRESHOLD = 90  # 3초 (30fps 기준)
RECORDING_BUFFER_FRAMES = 150    # 5초의 추가 녹화
MIN_RECORDING_FRAMES = 90        # 최소 3초 녹화
TARGET_FPS = 30                  # 목표 FPS
FRAME_DROP_THRESHOLD = 0.5       # 프레임 드롭 임계값 (80% 미만이면 경고)
VIDEO_CODEC = 'avc1'             # H.264 코덱 사용

# 도난 위험도에 따른 저장 경로
RISK_PATHS = {
    "danger": os.path.join(VIDEO_SAVE_PATH, "danger"),
    "warning": os.path.join(VIDEO_SAVE_PATH, "middle"),
    "safe": os.path.join(VIDEO_SAVE_PATH, "safety"),
    "other": os.path.join(VIDEO_SAVE_PATH, "other")
}

# 각 저장 경로가 없으면 생성
for path in RISK_PATHS.values():
    os.makedirs(path, exist_ok=True)

# ========================== 카메라 관리 변수 ===========================
CAMERA_CONFIG = {
    0: {"name": "CCTV-A002", "resolution": (640, 480)},  # 웹캠
    1: {"name": "CCTV-B003", "resolution": (640, 480)},  # 추가 카메라 1
    2: {"name": "CCTV-B001", "resolution": (640, 480)},  # 추가 카메라 2
    3: {"name": "CCTV-A003", "resolution": (640, 480)}   # 추가 카메라 3
}

camera_streams = {}  # 활성화된 카메라 스트림 저장
failed_cameras = set()  # 연결 실패한 카메라 ID 저장
camera_locks = {}  # 각 카메라별 스레드 락

def retry_failed_cameras():
    """실패한 카메라에 대한 재연결 시도"""
    retry_cameras = failed_cameras.copy()
    for camera_id in retry_cameras:
        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                width, height = CAMERA_CONFIG[camera_id]["resolution"]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                camera_streams[camera_id] = cap
                camera_locks[camera_id] = threading.Lock()
                failed_cameras.remove(camera_id)
                logger.info(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 재연결 성공")
            else:
                logger.warning(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 재연결 시도 실패")
        except Exception as e:
            logger.error(f"카메라 {camera_id} 재연결 시도 중 오류: {e}")

def check_camera_status():
    """모든 카메라의 상태를 확인하고 필요한 경우 재연결"""
    for camera_id in list(camera_streams.keys()):
        try:
            with camera_locks[camera_id]:
                cap = camera_streams[camera_id]
                if not cap.isOpened() or not cap.grab():
                    logger.warning(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 연결 끊김 감지")
                    cap.release()
                    del camera_streams[camera_id]
                    del camera_locks[camera_id]
                    failed_cameras.add(camera_id)
        except Exception as e:
            logger.error(f"카메라 {camera_id} 상태 확인 중 오류: {e}")
            failed_cameras.add(camera_id)

    if failed_cameras:
        retry_failed_cameras()

# 기존 initialize_cameras 함수 수정
def initialize_cameras():
    """사용 가능한 모든 카메라 초기화"""
    for camera_id in CAMERA_CONFIG.keys():
        try:
            if camera_id not in camera_streams and camera_id not in failed_cameras:
                cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                if cap.isOpened():
                    width, height = CAMERA_CONFIG[camera_id]["resolution"]
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    # 프레임 버퍼 초기화
                    cap.grab()
                    camera_streams[camera_id] = cap
                    camera_locks[camera_id] = threading.Lock()
                    logger.info(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 초기화 성공")
                else:
                    failed_cameras.add(camera_id)
                    logger.warning(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 초기화 실패")
        except Exception as e:
            failed_cameras.add(camera_id)
            logger.error(f"카메라 {camera_id} 초기화 중 오류: {e}")
    
    # 실패한 카메라에 대해 즉시 재시도
    if failed_cameras:
        retry_failed_cameras()

# 주기적인 카메라 상태 확인을 위한 스케줄러 설정
def start_camera_monitor():
    """카메라 모니터링 스케줄러 시작"""
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_camera_status, 'interval', seconds=30)
    scheduler.start()
    logger.info("카메라 모니터링 스케줄러 시작됨")

# YOLOv8 모델 로드
person_detector = YOLO('yolov8n.pt')
person_detector.conf = 0.50  # 신뢰도 임계값 설정
person_detector.verbose = False  # 출력 비활성화

# ========================== 보안 및 인증 ===========================
hashed_password = bcrypt.hashpw(b"1234", bcrypt.gensalt())

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

# ========================== 로그인 / 로그아웃 ===========================
@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if username == 'admin' and bcrypt.checkpw(password.encode('utf-8'), hashed_password):
        session['logged_in'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '아이디 또는 비밀번호가 올바르지 않습니다.'}), 401

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login_page'))

# ========================== 유틸리티 함수 ===========================
def play_alert_sound():
    """도난 감지 시 경고음 재생"""
    def play():
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(ALERT_SOUND_PATH)
            pygame.mixer.music.play()
            logger.info("경고음 재생")
        except Exception as e:
            logger.error(f"경고음 재생 실패: {e}")
    
    # 별도 스레드에서 실행하여 메인 스레드 블로킹 방지
    Thread(target=play, daemon=True).start()

# ========================== 유틸 함수 ===========================
def get_risk_level(steal_score):
    """도난 예측값에 따른 위험도 레벨 반환"""
    if steal_score >= HYPER_VALUE_STEAL:
        return "danger"
    elif steal_score >= HYPER_VALUE_WARNING:
        return "warning" # middle 대신 warning 사용
    else:
        return "safety" # safe 대신 safety 사용

def save_video_to_db(video_path, risk_level, steal_score, cctv_name="알 수 없음", duration=None):
    """비디오 파일을 GridFS에 저장하고 메타데이터를 해당 위험도 컬렉션에 저장"""
    try:
        logger.info(f"비디오 저장 시작: {video_path}, 위험도: {risk_level}")

        # video_db 클라이언트 사용
        fs = gridfs.GridFS(video_db)
        now = datetime.now()

        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            logger.error(f"비디오 파일이 존재하지 않음: {video_path}")
            return None
            
        # 비디오 파일 크기 확인
        file_size = os.path.getsize(video_path)
        logger.info(f"비디오 파일 크기: {file_size} bytes")
        
        # 파일명 생성
        filename = os.path.basename(video_path)
        # 상대 경로는 DB 저장 시 사용하지 않음
        
        # GridFS에 파일 저장
        file_id = fs.put(
            open(video_path, 'rb'), # 파일 객체를 직접 전달
            filename=filename,
            content_type='video/mp4',
            upload_date=now,
            created_date=now, # 생성일 필드 추가
            risk_level=risk_level,
            steal_score=float(steal_score) if steal_score is not None else None, # steal_score가 None일 경우 처리
            file_size=file_size,
            status="active",
            codec=VIDEO_CODEC,
            cctv_name=cctv_name,
            duration=duration
        )
        
        logger.info(f"GridFS 저장 완료 - ID: {file_id}")
        
        # 메타데이터를 해당 위험도 컬렉션에 저장 (save_to_data_video 데이터베이스 사용)
        collection_name = f"{risk_level}_videos"
        # 해당 컬렉션이 없으면 자동 생성됨
        video_db[collection_name].insert_one({
            "_id": file_id, # GridFS 파일 ID와 동일하게 설정
            "filename": filename,
            # "video_path": video_path, # DB에 파일 경로를 직접 저장하지 않음
            "risk_level": risk_level,
            "steal_score": float(steal_score) if steal_score is not None else None,
            "file_size": file_size,
            "created_date": now,
            "status": "active",
            "cctv_name": cctv_name,
            "duration": duration,
            "upload_date": now # 정렬을 위해 upload_date 필드 추가
        })
        
        logger.info(f"메타데이터 저장 완료 - 컬렉션: {collection_name}")
        
        # 파일 시스템의 임시 파일 삭제
        os.remove(video_path)
        logger.info(f"임시 파일 삭제 완료: {video_path}")
        
        return file_id
        
    except Exception as e:
        logger.error(f"DB 저장 실패: {e}")
        return None

def get_recent_videos(risk_level=None, limit=50, skip=0, sort_order=-1):
    """최근 저장된 비디오 목록 조회 (save_to_data_video 데이터베이스 사용)"""
    try:
        # video_db 클라이언트 사용
        # client = MongoClient('mongodb://localhost:27017/')
        # db = client['video_surveillance']
        fs = gridfs.GridFS(video_db)
        
        # 쿼리 조건 (fs.files 컬렉션 사용)
        query = {"status": "active"}
        if risk_level and risk_level != 'all': # 'all' 필터는 여기서 처리하지 않고 가져온 후 필터링
            query["risk_level"] = risk_level
        
        # fs.files 컬렉션에서 파일 목록 조회
        files = video_db.fs.files.find(
            query,
            {
                "_id": 1,
                "filename": 1,
                "upload_date": 1,
                "risk_level": 1,
                "steal_score": 1,
                "file_size": 1,
                "cctv_name": 1,
                "duration": 1
            }
        ).sort("upload_date", sort_order).skip(skip).limit(limit)
        
        videos = []
        for file in files:
            video_info = {
                "id": str(file["_id"]),
                "filename": file["filename"],
                "upload_date": file["upload_date"],
                "created_date": file["upload_date"].strftime("%Y-%m-%d"), # 날짜 형식 변경
                "created_time": file["upload_date"].strftime("%H:%M:%S"), # 시간 형식 변경
                "risk_level": file.get("risk_level", "unknown"),
                "steal_score": file.get("steal_score"),
                "file_size": file.get("file_size", 0),
                "cctv_name": file.get("cctv_name", "알 수 없음"),
                "duration": file.get("duration")
            }
            videos.append(video_info)
        
        # 전체 문서 수 조회
        total_count = video_db.fs.files.count_documents(query)
        
        return {
            "videos": videos,
            "total_count": total_count,
            "current_page": skip // limit + 1 if limit > 0 else 1,
            "total_pages": (total_count + limit - 1) // limit if limit > 0 else 1
        }
        
    except Exception as e:
        logger.error(f"비디오 목록 조회 실패: {e}")
        return {"videos": [], "total_count": 0, "current_page": 1, "total_pages": 1}

def get_video_stats():
    """비디오 통계 정보 조회 (save_to_data_video 데이터베이스의 각 위험도 컬렉션 사용)"""
    try:
        # video_db 클라이언트 사용
        # client = MongoClient('mongodb://localhost:27017/')
        # db = client['video_surveillance'] 

        stats = {
            'danger': {'count': 0, 'percent': 0, 'avg_score': 0},
            'warning': {'count': 0, 'percent': 0, 'avg_score': 0},
            'safety': {'count': 0, 'percent': 0, 'avg_score': 0}
        }

        total_active_count = 0

        # 각 위험도 컬렉션에서 active 상태 문서 수 계산
        for risk_level in ['danger', 'warning', 'safety']:
            collection_name = f"{risk_level}_videos"
            count = video_db[collection_name].count_documents({'status': 'active'})
            stats[risk_level]['count'] = count
            total_active_count += count

            # 평균 steal_score 계산 (선택 사항, 필요시 쿼리 추가)
            # pipeline = [{'$match': {'status': 'active'}}, {'$group': {'_id': None, 'avg_score': {'$avg': '$steal_score'}}}]
            # result = list(video_db[collection_name].aggregate(pipeline))
            # if result: stats[risk_level]['avg_score'] = result[0]['avg_score'] or 0

        # 백분율 계산
        for risk_level in ['danger', 'warning', 'safety']:
            if total_active_count > 0:
                stats[risk_level]['percent'] = (stats[risk_level]['count'] / total_active_count) * 100

        return stats # 딕셔너리 형태로 반환
    except Exception as e:
        logger.error(f"통계 정보 조회 실패: {e}")
        return {
            'danger': {'count': 0, 'percent': 0, 'avg_score': 0},
            'warning': {'count': 0, 'percent': 0, 'avg_score': 0},
            'safety': {'count': 0, 'percent': 0, 'avg_score': 0}
        }

def put_text_pil(frame, text, position, font_size=32, color=(255, 255, 255)):
    """PIL을 사용하여 한글 텍스트 그리기"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame_bgr

def draw_probability_bars(frame: np.ndarray, steal_prob: float, purchase_prob: float) -> np.ndarray:
    """확률을 시각적 프로그레스 바로 표시"""
    if frame is None or not isinstance(frame, np.ndarray):
        return frame
        
    height, width = frame.shape[:2]
    bar_height = 20
    bar_width = 200
    margin = 10
    
    # 배경 박스
    bg_color = (50, 50, 50)
    cv2.rectangle(frame, 
                 (margin, margin), 
                 (margin + bar_width + 100, margin + 2 * bar_height + 40),
                 bg_color, -1)
    
    # 물품습득 확률 바
    steal_width = int(bar_width * steal_prob)
    if steal_prob < 0.7:
        steal_color = (0, 255, 0)  # 초록
    elif steal_prob < 0.98:
        steal_color = (0, 255, 255)  # 노랑
    else:
        steal_color = (0, 0, 255)  # 빨강
        
    cv2.rectangle(frame, 
                 (margin + 100, margin + 5), 
                 (margin + 100 + bar_width, margin + bar_height),
                 (150, 150, 150), -1)  # 배경 바
    cv2.rectangle(frame, 
                 (margin + 100, margin + 5), 
                 (margin + 100 + steal_width, margin + bar_height),
                 steal_color, -1)  # 실제 확률 바
    
    # 구매 확률 바
    purchase_width = int(bar_width * purchase_prob)
    purchase_color = (0, 255, 0)  # 초록색
    
    cv2.rectangle(frame, 
                 (margin + 100, margin + bar_height + 15), 
                 (margin + 100 + bar_width, margin + 2 * bar_height + 10),
                 (150, 150, 150), -1)  # 배경 바
    cv2.rectangle(frame, 
                 (margin + 100, margin + bar_height + 15), 
                 (margin + 100 + purchase_width, margin + 2 * bar_height + 10),
                 purchase_color, -1)  # 실제 확률 바
    
    # 텍스트 추가
    frame = put_text_pil(frame, "물품습득:", (margin + 10, margin + bar_height), 
                        font_size=15, color=(255, 255, 255))
    frame = put_text_pil(frame, "구매:", (margin + 10, margin + 2 * bar_height + 5), 
                        font_size=15, color=(255, 255, 255))
    
    # 확률 수치 표시
    frame = put_text_pil(frame, f"{steal_prob:.1%}", 
                        (margin + bar_width + 110, margin + bar_height), 
                        font_size=15, color=steal_color)
    frame = put_text_pil(frame, f"{purchase_prob:.1%}", 
                        (margin + bar_width + 110, margin + 2 * bar_height + 5), 
                        font_size=15, color=purchase_color)
    
    return frame

# ========================== 비디오 스트리밍 ===========================
def generate_video_stream(camera_id=0):
    """비디오 스트림 생성"""
    if camera_id not in CAMERA_CONFIG:
        logger.error(f"잘못된 카메라 ID: {camera_id}")
        return None

    cap = get_camera_stream(camera_id)
    if cap is None:
        return generate_error_stream(f"카메라 {CAMERA_CONFIG[camera_id]['name']} 연결 실패")

    # 프레임 처리 통계 초기화
    frame_count = 0
    start_time = time.time()
    processed_frames = 0
    last_fps_check = start_time
    fps_history = deque(maxlen=30)  # 최근 30개의 FPS 기록
    avg_fps = 0.0  # FPS 초기값 설정

    frames = deque(maxlen=SEQUENCE_LENGTH)
    recording = False
    has_abnormal_behavior = False
    max_steal_score = 0
    max_purchase_score = 0
    no_person_frames = 0
    video_writer = None
    temp_video_path = None

    # 초기 예측값 설정
    preds_steal = 0.0
    preds_buy = 0.0

    while True:
        try:
            with camera_locks[camera_id]:
                success, frame = cap.read()
                if not success:
                    logger.error(f"카메라 {CAMERA_CONFIG[camera_id]['name']} 프레임 읽기 실패")
                    return generate_error_stream(f"카메라 {CAMERA_CONFIG[camera_id]['name']} 송출 불가")

            frame_count += 1
            current_time = time.time()
            
            # FPS 모니터링 (1초마다 체크)
            if current_time - last_fps_check >= 1.0:
                current_fps = processed_frames / (current_time - last_fps_check)
                fps_history.append(current_fps)
                avg_fps = sum(fps_history) / len(fps_history)
                
                if avg_fps < TARGET_FPS * FRAME_DROP_THRESHOLD:
                    logger.warning(f"프레임 드롭 발생 (현재 FPS: {avg_fps:.1f}, 목표: {TARGET_FPS})")
                
                processed_frames = 0
                last_fps_check = current_time

            display_frame = frame.copy()
            height, width = frame.shape[:2]  # 프레임 크기 가져오기

            # YOLO로 사람 감지
            results = person_detector(frame, classes=[0], stream=True)
            person_in_frame = False
            
            for result in results:
                if len(result.boxes) > 0:
                    person_in_frame = True
                    # 감지된 사람 박스 그리기
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        if conf > 0.4:  # 신뢰도가 0.4 이상인 경우만 표시
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            conf_text = f"Person: {conf:.2f}"
                            cv2.putText(display_frame, conf_text, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break

            if not person_in_frame:
                no_person_frames += 1
            else:
                no_person_frames = 0

            frames.append(frame)
            processed_frames += 1

            if len(frames) == SEQUENCE_LENGTH:
                try:
                    # 행동 분석 (출력 숨기기)
                    with suppress_stdout():
                        results_abnormal = model_abnormal(inputs=np.array(frames))
                        results_purchase = model_purchase(inputs=np.array(frames))
                    
                    preds_none, preds_steal = results_abnormal["predictions"][0]["rec_scores"][0]
                    rec_scores = results_purchase["predictions"][0]["rec_scores"][0]
                    preds_buy = rec_scores[1]

                    # 최대 확률 업데이트
                    max_steal_score = max(max_steal_score, preds_steal)
                    max_purchase_score = max(max_purchase_score, preds_buy)

                    # 확률 바 추가
                    display_frame = draw_probability_bars(display_frame, preds_steal, preds_buy)
                except Exception as e:
                    logger.error(f"행동 분석 중 오류 발생: {str(e)}")
                    continue

            # FPS 표시
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (width - 150, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 녹화 시작 조건 (사람만 감지되면 녹화)
            if person_in_frame and not recording:
                recording = True
                no_person_frames = 0
                now = datetime.now()
                temp_video_path = os.path.join(RISK_PATHS["other"], 
                                             f"temp_event_{now.strftime('%Y%m%d_%H%M%S')}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
                video_writer = cv2.VideoWriter(temp_video_path, 
                                             fourcc,
                                             fps, 
                                             (width, height))
                play_alert_sound()
                logger.info(f"사람 감지로 녹화 시작 - 임시 저장 위치: {temp_video_path}")

            # 녹화 중 처리 (모든 프레임 저장)
            if recording:
                video_writer.write(frame)
                # 이상행동 감지
                if len(frames) == SEQUENCE_LENGTH:
                    if preds_steal > HYPER_VALUE_STEAL and not has_abnormal_behavior:
                        has_abnormal_behavior = True
                        max_steal_score = preds_steal
                        logger.info(f"이상행동 감지! (물품습득득 확률: {preds_steal:.2%})")
                    # 구매 행동 확률 업데이트
                    if has_abnormal_behavior:
                        max_purchase_score = max(max_purchase_score, preds_buy)

            # 사람이 나가면 녹화 종료 및 처리
            if no_person_frames >= NO_PERSON_FRAMES_THRESHOLD and recording:
                recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                    
                    if has_abnormal_behavior:
                        # 구매 행동 확률에 따른 처리
                        if max_purchase_score >= 0.8:  # 구매 확률 80% 이상
                            risk_level = "safe"
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            new_filename = f"purchase_event_{timestamp}.mp4"
                            new_path = os.path.join(RISK_PATHS[risk_level], new_filename)
                            
                            if os.path.exists(temp_video_path):
                                shutil.move(temp_video_path, new_path)
                                save_video_to_db(new_path, risk_level, max_steal_score)
                                logger.info(f"구매 행동으로 판단되어 안전 영상 저장됨 (구매 확률: {max_purchase_score:.2%})")

                        else:
                            if max_purchase_score >= 0 and max_purchase_score < 0.8:  # 구매 확률 70% 미만
                                if max_steal_score >=0.97:
                                    risk_level = "danger"
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    new_filename = f"theft_event_{timestamp}.mp4"
                                    new_path = os.path.join(RISK_PATHS[risk_level], new_filename)
                                
                                    if os.path.exists(temp_video_path):
                                        shutil.move(temp_video_path, new_path)
                                        save_video_to_db(new_path, risk_level, max_steal_score)
                                        play_alert_sound()
                                        send_kakao_message(f"물품습득 의심 행동 감지! (물품습득 확률: {max_steal_score:.2%}, 구매 확률: {max_purchase_score:.2%})")
                                        logger.info(f"도난 의심 행동으로 판단되어 위험 영상 저장됨")

                                else:
                                    risk_level = "warning"
                                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    new_filename = f"suspicious_event_{timestamp}.mp4"
                                    new_path = os.path.join(RISK_PATHS[risk_level], new_filename)
                                    
                                    if os.path.exists(temp_video_path):
                                        shutil.move(temp_video_path, new_path)
                                        save_video_to_db(new_path, risk_level, max_steal_score)
                                        play_alert_sound()
                                        send_kakao_message(f"의심 행동 감지! (물품습득 확률: {max_steal_score:.2%}, 구매 확률: {max_purchase_score:.2%})")
                                        logger.info(f"의심 행동으로 판단되어 중간 위험도 영상 저장됨")
                            
                    else:
                        # 이상행동이 없었던 경우 임시 파일 삭제
                        if os.path.exists(temp_video_path):
                            os.remove(temp_video_path)
                            logger.info("정상 행동으로 판단되어 임시 영상 삭제됨")

                    # 변수 초기화
                    has_abnormal_behavior = False
                    max_steal_score = 0
                    max_purchase_score = 0
                    frames.clear()

            # 프레임 출력
            _, buffer = cv2.imencode('.jpg', display_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + 
                    buffer.tobytes() + b'\r\n')

        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            continue

    # 최종 FPS 통계 출력
    total_time = time.time() - start_time
    final_fps = frame_count / total_time
    logger.info(f"\n=== 처리 통계 ===")
    logger.info(f"총 프레임 수: {frame_count}")
    logger.info(f"처리 시간: {total_time:.1f}초")
    logger.info(f"평균 FPS: {final_fps:.1f}")
    logger.info(f"프레임 드롭률: {((TARGET_FPS - final_fps) / TARGET_FPS * 100):.1f}%")
    logger.info("================\n")

# ========================== rich 진행바 subprocess 예제 ===========================
@app.route('/run-rich-job')
def run_rich_job():
    def generate():
        try:
            cmd = ["python", "rich_job.py"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    yield f"data: {line.strip()}\n\n"
                    
            process.stdout.close()
            process.wait()
            
        except Exception as e:
            print(f"Rich job 실행 중 오류 발생: {str(e)}")
            yield f"data: Error: {str(e)}\n\n"

    return Response(stream_with_context(generate()), 
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'X-Accel-Buffering': 'no'
                   })

# ========================== 라우팅 ===========================

def get_camera_stream(camera_id):
    """특정 카메라의 스트림 가져오기"""
    if camera_id in failed_cameras:
        # 실패한 카메라에 대해 재연결 시도
        retry_failed_cameras()
        if camera_id in failed_cameras:
            return None
    
    if camera_id not in camera_streams:
        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                width, height = CAMERA_CONFIG[camera_id]["resolution"]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                # 프레임 버퍼 초기화
                cap.grab()
                camera_streams[camera_id] = cap
                camera_locks[camera_id] = threading.Lock()
                logger.info(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 연결 성공")
                return cap
            else:
                failed_cameras.add(camera_id)
                logger.warning(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 연결 실패")
                return None
        except Exception as e:
            failed_cameras.add(camera_id)
            logger.error(f"카메라 {camera_id} 연결 중 오류: {e}")
            return None
    
    # 카메라 상태 확인
    try:
        with camera_locks[camera_id]:
            cap = camera_streams[camera_id]
            if not cap.isOpened() or not cap.grab():
                logger.warning(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 연결 끊김 감지")
                cap.release()
                del camera_streams[camera_id]
                del camera_locks[camera_id]
                failed_cameras.add(camera_id)
    except Exception as e:
        logger.error(f"카메라 {camera_id} 상태 확인 중 오류: {e}")
        failed_cameras.add(camera_id)
        return None
    
    return camera_streams[camera_id]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
@login_required
def main_page():
    try:
        # 영상 데이터베이스 클라이언트 사용 (전역 video_db 객체 사용)
        # client = MongoClient('mongodb://localhost:27017/') # 필요 없습니다.
        # db = client['save_to_data_video'] # 필요 없습니다.

        # 각 카테고리별 전체 개수 가져오기 (save_to_data_video 데이터베이스 사용)
        danger_count = video_db['danger_videos'].count_documents({'status': 'active'})
        middle_count = video_db['warning_videos'].count_documents({'status': 'active'}) # warning_videos에서 count
        safety_count = video_db['safety_videos'].count_documents({'status': 'active'})

        # 전체 개수 계산
        total_count = danger_count + middle_count + safety_count

        # 백분율 계산 (전체가 0인 경우 처리)
        danger_percent = (danger_count / total_count * 100) if total_count > 0 else 0
        middle_percent = (middle_count / total_count * 100) if total_count > 0 else 0
        safety_percent = (safety_count / total_count * 100) if total_count > 0 else 0

        # 통계 딕셔너리 구성 (템플릿이 예상하는 middle 키 사용)
        stats = {
            'danger': {'count': danger_count, 'percent': danger_percent},
            'middle': {'count': middle_count, 'percent': middle_percent}, # warning 대신 middle 키 사용
            'safety': {'count': safety_count, 'percent': safety_percent}
        }

        # 최근 이벤트 3개 가져오기 (danger와 warning 컬렉션에서 가져와 병합 후 정렬)
        # main 페이지의 사고 이벤트 목록은 danger와 warning만 표시하는 것 같으므로 해당 컬렉션에서 가져옴
        recent_danger_videos = list(video_db['danger_videos'].find({'status': 'active'}).sort('created_date', -1).limit(3))
        recent_warning_videos = list(video_db['warning_videos'].find({'status': 'active'}).sort('created_date', -1).limit(3))

        recent_events = []

        # danger 영상 처리
        for video in recent_danger_videos:
             recent_events.append({
                 "id": str(video["_id"]),
                 "risk_level": video.get("risk_level", "danger"),
                 "badge_class": "danger",
                 "event_type": "도난",
                 "filename": video.get("filename", "알 수 없음"),
                 "created_date": video.get("created_date", datetime.min).strftime("%Y-%m-%d") if isinstance(video.get("created_date"), datetime) else "N/A",
                 "created_time": video.get("created_date", datetime.min).strftime("%H:%M:%S") if isinstance(video.get("created_date"), datetime) else "N/A",
                 "video_url": url_for('serve_video', file_id=str(video["_id"])),
                 "steal_score": video.get("steal_score"),
             })

        # warning 영상 처리
        for video in recent_warning_videos:
             recent_events.append({
                 "id": str(video["_id"]),
                 "risk_level": video.get("risk_level", "warning"),
                 "badge_class": "warning",
                 "event_type": "의심",
                 "filename": video.get("filename", "알 수 없음"),
                 "created_date": video.get("created_date", datetime.min).strftime("%Y-%m-%d") if isinstance(video.get("created_date"), datetime) else "N/A",
                 "created_time": video.get("created_date", datetime.min).strftime("%H:%M:%S") if isinstance(video.get("created_date"), datetime) else "N/A",
                 "video_url": url_for('serve_video', file_id=str(video["_id"])),
                 "steal_score": video.get("steal_score"),
             })

        # 시간순으로 최종 정렬하고 상위 3개만 선택
        recent_events.sort(key=lambda x: (x.get('created_date', '0000-00-00'), x.get('created_time', '00:00:00')), reverse=True)
        recent_events = recent_events[:3]

        # CCTV 연결 상태 카운트 계산
        connected_count = len(camera_streams) if 'camera_streams' in globals() else 0
        disconnected_count = len(CAMERA_CONFIG) - connected_count if 'CAMERA_CONFIG' in globals() else 0

        return render_template('main.html',
                           active_tab='dashboard',
                           connected_count=connected_count,
                           disconnected_count=disconnected_count,
                           recent_events=recent_events,
                           stats=stats)

    except Exception as e:
        logger.error(f"메인 페이지 로딩 실패: {e}")
        # 에러 발생 시 빈 데이터 및 기본값과 함께 템플릿 렌더링
        return render_template('main.html',
                           active_tab='dashboard',
                           connected_count=len(camera_streams) if 'camera_streams' in globals() else 0, # 에러 시 기본값 계산
                           disconnected_count=len(CAMERA_CONFIG) - (len(camera_streams) if 'camera_streams' in globals() else 0) if 'CAMERA_CONFIG' in globals() else 0, # 에러 시 기본값 계산
                           recent_events=[],
                           stats={
                                'danger': {'count': 0, 'percent': 0},
                                'middle': {'count': 0, 'percent': 0},
                                'safety': {'count': 0, 'percent': 0}
                           })

@app.route('/recodingvideos')
@login_required
def recorded_videos_page():
    try:
        # 1. 통계 데이터 가져오기 (main과 동일 구조)
        stats_data_from_db = get_video_stats()
        stats = {
            'danger': stats_data_from_db.get('danger', {'count': 0, 'percent': 0, 'avg_score': 0}),
            'middle': stats_data_from_db.get('warning', {'count': 0, 'percent': 0, 'avg_score': 0}),  # warning -> middle
            'safety': stats_data_from_db.get('safety', {'count': 0, 'percent': 0, 'avg_score': 0}),
        }
        # 2. 백분율 재계산 (main과 동일)
        total_count = stats['danger']['count'] + stats['middle']['count'] + stats['safety']['count']
        if total_count > 0:
            stats['danger']['percent'] = (stats['danger']['count'] / total_count) * 100
            stats['middle']['percent'] = (stats['middle']['count'] / total_count) * 100
            stats['safety']['percent'] = (stats['safety']['count'] / total_count) * 100
        # 3. 최근 이벤트 3개 가져오기
        recent_events_data = get_recent_videos(limit=3, risk_level=None)
        recent_events = recent_events_data.get('videos', [])
        # 4. 카메라 연결 상태 계산
        connected_count = len(camera_streams)
        disconnected_count = len(CAMERA_CONFIG) - connected_count

        return render_template(
            'recodingvideos.html',
            active_tab='recorded_videos',
            connected_count=connected_count,
            disconnected_count=disconnected_count,
            recent_events=recent_events,
            stats=stats
        )
    except Exception as e:
        logger.error(f"녹화 영상 페이지 로딩 실패: {e}")
        # 오류시 main과 동일하게 빈 값 반환
        stats_on_error = {
            'danger': {'count': 0, 'percent': 0, 'avg_score': 0},
            'middle': {'count': 0, 'percent': 0, 'avg_score': 0},
            'safety': {'count': 0, 'percent': 0, 'avg_score': 0}
        }
        return render_template(
            'recodingvideos.html',
            active_tab='recorded_videos',
            connected_count=len(camera_streams) if 'camera_streams' in globals() else 0,
            disconnected_count=len(CAMERA_CONFIG) - (len(camera_streams) if 'camera_streams' in globals() else 0) if 'CAMERA_CONFIG' in globals() else 0,
            stats=stats_on_error,
            recent_events=[]
        )


@app.route('/statistics')
@login_required
def statistics_page():
    return render_template('statistics.html')

@app.route('/video_feed/<int:camera_id>', methods=['GET', 'HEAD'])
@login_required
def video_feed(camera_id):
    if camera_id in failed_cameras:
        if request.method == 'HEAD':
            return Response(status=404)
        return Response(generate_error_stream(f"카메라 {camera_id}는 연결할 수 없습니다.\n다른 카메라를 이용해주세요."), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    if request.method == 'HEAD':
        cap = get_camera_stream(camera_id)
        if not cap:
            return Response(status=404)
        ret, _ = cap.read()
        if not ret:
            return Response(status=404)
        return Response(status=200)

    stream = generate_video_stream(camera_id)
    if stream is None:
        return Response(status=404)
    return Response(stream, mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_error_stream(error_message):
    """연결 불가능한 카메라에 대한 에러 메시지 스트림 생성"""
    while True:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, error_message, (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                buffer.tobytes() + b'\r\n')
        time.sleep(0.5)

@app.route('/video/<file_id>')
@login_required
def serve_video(file_id):
    """컬렉션과 GridFS 연동 비디오 스트리밍 (save_to_data_video 데이터베이스 사용)"""
    try:
        logger.info(f"비디오 스트리밍 요청: {file_id}")

        # video_db 클라이언트 사용
        fs = gridfs.GridFS(video_db)
        gridfs_id = None

        try:
            # 1. 모든 위험도 컬렉션에서 문서 _id로 먼저 찾음
            file_obj_id = ObjectId(file_id)
            video_doc = None
            for risk_level in ['danger', 'warning', 'safety', 'other']:
                 collection_name = f"{risk_level}_videos"
                 video_doc = video_db[collection_name].find_one({"_id": file_obj_id})
                 if video_doc: break # 찾으면 루프 종료

            # 컬렉션 문서에서 GridFS ID를 가져오거나, _id 자체를 GridFS ID로 사용
            if video_doc:
                gridfs_id = video_doc.get("gridfs_id", file_obj_id) # gridfs_id 필드가 없으면 _id 사용
            else:
                 # 컬렉션에서 못 찾았으면 file_id 자체가 GridFS의 ObjectId라고 가정
                 gridfs_id = file_obj_id

        except Exception:
            # file_id가 ObjectId 형식이 아니거나 처리 중 예외 발생 시
            logger.error(f"비디오 ID ({file_id}) 처리 중 오류 또는 잘못된 형식")
            return "Invalid video ID or processing error", 400

        # 2. GridFS에 해당 파일 존재 여부 확인 (video_db 사용)
        if not fs.exists(gridfs_id):
            logger.error(f"GridFS에서 비디오 파일을 찾을 수 없음: {gridfs_id}")
            return "Video not found", 404

        # 3. GridFS에서 파일 읽기 (스트리밍) (video_db 사용)
        video_file = fs.get(gridfs_id)
        logger.info(f"비디오 파일 정보: {video_file.filename}, 크기: {video_file.length} bytes")

        # 4. 비디오 스트리밍 (chunk 단위로 클라이언트에 전달)
        def generate():
            chunk_size = 1024 * 1024  # 1MB씩 전송
            data = video_file.read(chunk_size)
            while data:
                yield data
                data = video_file.read(chunk_size)

        # 5. Flask Response로 반환
        return Response(
            generate(),
            mimetype='video/mp4',
            headers={
                'Content-Disposition': f'inline; filename={video_file.filename}',
                'Content-Type': 'video/mp4',
                'Content-Length': str(video_file.length),
                 'Accept-Ranges': 'bytes' # Range 헤더 지원 추가
            }
        )

    except Exception as e:
        logger.error(f"비디오 스트리밍 실패: {e}")
        return "Error streaming video", 500

def delete_video(file_id):
    """비디오 파일 삭제 (save_to_data_video 데이터베이스의 GridFS 및 컬렉션 업데이트)"""
    try:
        # video_db 클라이언트 사용
        # client = MongoClient('mongodb://localhost:27017/') # 필요 없습니다.
        # db = client['video_surveillance'] # 필요 없습니다.

        # GridFS 파일 상태 업데이트
        result_fs = video_db.fs.files.update_one(
            {"_id": ObjectId(file_id)},
            {
                "$set": {
                    "status": "deleted",
                    "deleted_at": datetime.now()
                }
            }
        )
        
        # 해당 위험도 컬렉션 문서 상태 업데이트
        # 어떤 컬렉션에 속하는지 모르므로 모든 위험도 컬렉션에서 시도
        modified_count_collection = 0
        for risk_level in ['danger', 'warning', 'safety', 'other']:
             collection_name = f"{risk_level}_videos"
             result_col = video_db[collection_name].update_one(
                 {"_id": ObjectId(file_id)},
                 {
                     "$set": {
                         "status": "deleted",
                         "deleted_at": datetime.now()
                     }
                 }
             )
             modified_count_collection += result_col.modified_count

        logger.info(f"비디오 삭제 처리 완료 - ID: {file_id}, fs.files 업데이트: {result_fs.modified_count}, 컬렉션 업데이트: {modified_count_collection}")
        
        return result_fs.modified_count > 0 or modified_count_collection > 0

    except Exception as e:
        logger.error(f"비디오 삭제 실패: {e}")
        return False

@app.route('/api/videos')
@login_required
def get_videos_api():
    """저장된 비디오 목록을 위험도 별로 반환 (save_to_data_video 데이터베이스 사용)"""
    try:
        # video_db 클라이언트 사용
        # client = MongoClient('mongodb://localhost:27017/') 
        # db = client['video_surveillance']
        risk_level = request.args.get('risk_level', 'all')
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        sort_order = -1 if request.args.get('sort', 'desc').lower() == 'desc' else 1

        skip = (page - 1) * per_page

        # 쿼리 조건
        query = {'status': 'active'}
        collection_name = None
        if risk_level != 'all':
            # 특정 위험도의 경우 해당 컬렉션만 조회
            collection_name = f"{risk_level}_videos"
        # else: all의 경우는 아래에서 모든 컬렉션 조회

        videos = []
        total_count = 0

        if collection_name: # 특정 위험도 필터링
            if collection_name in video_db.list_collection_names(): # 컬렉션이 존재하는지 확인
                 videos = list(video_db[collection_name].find(query).sort('created_date', sort_order).skip(skip).limit(per_page))
                 total_count = video_db[collection_name].count_documents(query)
        else: # 전체 보기 (all)
             # 모든 위험도 컬렉션에서 active 상태 문서 가져와 병합
             all_videos = []
             for level in ['danger', 'warning', 'safety', 'other']:
                 col_name = f"{level}_videos"
                 if col_name in video_db.list_collection_names(): # 컬렉션이 존재하는지 확인
                      all_videos.extend(list(video_db[col_name].find({'status': 'active'})))
                      total_count += video_db[col_name].count_documents({'status': 'active'})

             # 시간순 정렬 후 페이지네이션 적용
             all_videos.sort(key=lambda x: x.get('created_date', datetime.min), reverse=True)
             videos = all_videos[skip:skip + per_page] # 슬라이싱으로 페이지네이션

        # 응답 데이터 구성
        video_list = []
        for video in videos:
            video_data = {
                "id": str(video["_id"]),
                "risk_level": video.get("risk_level", "unknown"),
                "filename": video.get("filename", "알 수 없음"),
                "created_date": video.get("created_date", datetime.min).strftime("%Y-%m-%d") if isinstance(video.get("created_date"), datetime) else "N/A",
                "created_time": video.get("created_date", datetime.min).strftime("%H:%M:%S") if isinstance(video.get("created_date"), datetime) else "N/A",
                "file_size": video.get("file_size", 0),
                "steal_score": video.get("steal_score"),
                "cctv_name": video.get("cctv_name", "알 수 없음"),
                "duration": video.get("duration"),
                "video_url": url_for('serve_video', file_id=str(video["_id"])),
                "timestamp": video.get('created_date', datetime.min).isoformat() # timestamp 필드 추가
            }
            video_list.append(video_data)

        response = {
            "videos": video_list,
            "pagination": {
                "total_items": total_count,
                "total_pages": (total_count + per_page - 1) // per_page if per_page > 0 else 1,
                "current_page": page,
                "per_page": per_page
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"API 오류: {e}")
        return jsonify({"error": "비디오 목록 조회 중 오류가 발생했습니다.", "details": str(e)}), 500

@app.route('/api/latest-videos')
@login_required
def get_latest_videos_api():
    """최신 영상 목록만 반환 (간단한 버전, save_to_data_video 데이터베이스 사용)"""
    try:
        # video_db 클라이언트 사용
        # client = MongoClient('mongodb://localhost:27017/') # 필요 없습니다.
        # db = client['video_surveillance'] # 필요 없습니다.

        # 모든 위험도 컬렉션에서 active 상태의 최신 10개 문서 가져와 병합 후 시간순 정렬
        videos = []
        for level in ['danger', 'warning', 'safety', 'other']:
             col_name = f"{level}_videos"
             videos.extend(list(video_db[col_name].find({'status': 'active'}).sort('created_date', -1).limit(10)))

        # 시간순으로 최종 정렬하고 상위 10개만 선택
        videos.sort(key=lambda x: x.get('created_date', datetime.min), reverse=True)
        videos = videos[:10]

        video_list = []
        for video in videos:
            video_data = {
                 "id": str(video["_id"]),
                 "risk_level": video.get("risk_level", "unknown"),
                 "filename": video.get("filename", "알 수 없음"),
                 "created_date": video.get("created_date", datetime.min).strftime("%Y-%m-%d") if isinstance(video.get("created_date"), datetime) else "N/A",
                 "created_time": video.get("created_date", datetime.min).strftime("%H:%M:%S") if isinstance(video.get("created_date"), datetime) else "N/A",
                 "cctv_name": video.get("cctv_name", "알 수 없음"),
                 "video_url": url_for('serve_video', file_id=str(video["_id"]))
            }
            video_list.append(video_data)

        return jsonify(video_list)

    except Exception as e:
        logger.error(f"최신 영상 조회 실패: {e}")
        return jsonify([])

@app.route('/api/stats')
@login_required
def get_stats_api():
    """비디오 통계 정보 API (get_video_stats 함수 사용)"""
    stats = get_video_stats() # 이미 save_to_data_video를 사용하도록 수정됨
    return jsonify(stats)

@app.route('/api/stream-url')
@login_required
def get_stream_url():
    camera_id = request.args.get('camera_id')
    stream_urls = {
        '1': 'http://yourserver.com/stream1',
        '2': 'http://yourserver.com/stream2',
        '3': 'http://yourserver.com/stream3',
        '4': 'http://yourserver.com/stream4',
    }
    return jsonify({'url': stream_urls.get(camera_id, '')})

@app.route('/api/visualizations')
@login_required
def get_visualization_images():
    try:
        image_dir = os.path.join('temp', 'vis_data')
        images = []
        if os.path.exists(image_dir):
            for fname in sorted(os.listdir(image_dir), reverse=True):
                if fname.lower().endswith(('.png', '.jpg')):
                    images.append(url_for('static', filename=os.path.join('..', image_dir, fname)))
        return jsonify(images[:10])
    except Exception as e:
        print(f"시각화 이미지 조회 오류: {e}")
        return jsonify([])

@app.route('/api/camera-status')
@login_required
def get_camera_status():
    """모든 카메라의 현재 상태 반환"""
    status = {}
    for camera_id in CAMERA_CONFIG.keys():
        camera_info = {
            'name': CAMERA_CONFIG[camera_id]['name'],
            'resolution': CAMERA_CONFIG[camera_id]['resolution'],
            'status': 'disconnected'
        }
        
        if camera_id in camera_streams:
            try:
                with camera_locks[camera_id]:
                    cap = camera_streams[camera_id]
                    if cap.isOpened() and cap.grab():
                        camera_info['status'] = 'connected'
                    else:
                        camera_info['status'] = 'error'
            except Exception:
                camera_info['status'] = 'error'
        elif camera_id in failed_cameras:
            camera_info['status'] = 'failed'
        
        status[camera_id] = camera_info
    
    return jsonify(status)

@app.route('/api/camera-reconnect/<int:camera_id>')
@login_required
def reconnect_camera(camera_id):
    """특정 카메라 재연결 시도"""
    if camera_id not in CAMERA_CONFIG:
        return jsonify({'success': False, 'message': '잘못된 카메라 ID'}), 400
    
    if camera_id in camera_streams:
        try:
            with camera_locks[camera_id]:
                cap = camera_streams[camera_id]
                if cap.isOpened():
                    cap.release()
                del camera_streams[camera_id]
                del camera_locks[camera_id]
        except Exception as e:
            logger.error(f"카메라 {camera_id} 연결 해제 중 오류: {e}")
    
    if camera_id in failed_cameras:
        failed_cameras.remove(camera_id)
    
    try:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            width, height = CAMERA_CONFIG[camera_id]["resolution"]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.grab()
            camera_streams[camera_id] = cap
            camera_locks[camera_id] = threading.Lock()
            logger.info(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 수동 재연결 성공")
            return jsonify({'success': True, 'message': '카메라 재연결 성공'})
        else:
            failed_cameras.add(camera_id)
            logger.warning(f"카메라 {CAMERA_CONFIG[camera_id]['name']} (ID: {camera_id}) 수동 재연결 실패")
            return jsonify({'success': False, 'message': '카메라 재연결 실패'}), 500
    except Exception as e:
        failed_cameras.add(camera_id)
        logger.error(f"카메라 {camera_id} 수동 재연결 중 오류: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ========================== 자동 삭제 관련 함수 ===========================
def get_video_datetime(filename):
    """파일명에서 날짜시간 추출"""
    try:
        # 파일명 예: temp_event_20250516_023442.mp4
        base = os.path.splitext(filename)[0]
        date_str = base.split('_')[2]  # '20250516'
        time_str = base.split('_')[3]  # '023442'
        return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
    except Exception as e:
        logger.warning(f"파일명 파싱 실패 ({filename}): {e}")
        return None

def cleanup_old_safety_videos():
    """24시간 이상 된 safety 영상 삭제 (파일과 DB 모두)"""
    safety_path = RISK_PATHS.get("safe", os.path.join(VIDEO_SAVE_PATH, "safety"))
    if not os.path.exists(safety_path):
        logger.warning(f"Safety 디렉토리가 존재하지 않음: {safety_path}")
        return

    # DB 연결 (필요시 글로벌 client/db 사용)
    client = MongoClient('mongodb://localhost:27017/')
    db = client['save_to_data_video']
    safety_collection = db['safety_videos']

    now = datetime.now()
    deleted_count = 0
    error_count = 0

    for filename in os.listdir(safety_path):
        if not filename.endswith('.mp4'):
            continue
        file_path = os.path.join(safety_path, filename)
        if not os.path.isfile(file_path):
            continue

        file_datetime = get_video_datetime(filename)
        if file_datetime is None:
            file_datetime = datetime.fromtimestamp(os.path.getctime(file_path))

        # 24시간 이상 경과
        if (now - file_datetime) > timedelta(days=1):
            try:
                os.remove(file_path)
                deleted_count += 1
                logger.info(f"오래된 영상 삭제 완료: {filename}")

                # === DB에서 soft delete ===
                relative_path = os.path.relpath(file_path, 'static').replace('\\', '/')
                result = safety_collection.update_one(
                    {"filename": filename},    # 또는 "video_path": relative_path
                    {
                        "$set": {
                            "status": "deleted",
                            "deleted_at": now
                        }
                    }
                )
                if result.modified_count == 1:
                    logger.info(f"DB 상태도 삭제 처리: {filename}")
                else:
                    logger.warning(f"DB 문서가 존재하지 않음 (파일만 삭제됨): {filename}")
            except Exception as e:
                error_count += 1
                logger.error(f"영상 삭제 실패 ({filename}): {e}")

    logger.info(f"정리 완료 - 삭제: {deleted_count}개, 실패: {error_count}개")

# 스케줄러 초기화 및 작업 등록
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=cleanup_old_safety_videos,
    trigger='interval',
    hours=1,
    id='cleanup_safety_videos'
)

# setup_db 실행을 위한 함수
def run_setup_db():
    try:
        python_executable = os.path.join(os.getcwd(), 'venv310', 'Scripts', 'python.exe')
        if not os.path.exists(python_executable):
            python_executable = 'python'  # 가상환경이 없으면 시스템 Python 사용
            
        subprocess.run([python_executable, 'setup_db.py'], check=True)
        logger.info("setup_db.py 실행 완료")
    except subprocess.CalledProcessError as e:
        logger.error(f"setup_db.py 실행 실패: {e}")
    except Exception as e:
        logger.error(f"setup_db.py 실행 중 오류 발생: {e}")

# setup_db 실행 작업 추가 (1분 간격)
scheduler.add_job(
    func=run_setup_db,
    trigger='interval',
    minutes=1,
    id='run_setup_db'
)

def setup_camera_system():
    initialize_cameras()
    start_camera_monitor()

@app.route('/api/filtered-videos')
@login_required
def get_filtered_videos():
    """저장된 비디오 목록을 위험도 별로 반환 (save_to_data_video 데이터베이스 사용)"""
    try:
        # video_db 클라이언트 사용
        # client = MongoClient('mongodb://localhost:27017/')
        # db = client['video_surveillance']
        risk_level = request.args.get('risk_level', 'all')
        # 페이지네이션 관련 파라미터는 이 API에서 직접 처리
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 5)) # 기본값을 5로 변경 (recodingvideos.html 필터 버튼 기준)
        sort_order = -1 if request.args.get('sort', 'desc').lower() == 'desc' else 1

        skip = (page - 1) * per_page

        # 쿼리 조건
        query = {'status': 'active'}
        collection_name = None
        if risk_level != 'all':
            # 특정 위험도의 경우 해당 컬렉션만 조회
            collection_name = f"{risk_level}_videos"
        # else: all의 경우는 아래에서 모든 컬렉션 조회

        videos = []
        total_count = 0

        if collection_name: # 특정 위험도 필터링
            if collection_name in video_db.list_collection_names(): # 컬렉션이 존재하는지 확인
                 videos = list(video_db[collection_name].find(query).sort('created_date', sort_order).skip(skip).limit(per_page))
                 total_count = video_db[collection_name].count_documents(query)
        else: # 전체 보기 (all)
             # 모든 위험도 컬렉션에서 active 상태 문서 가져와 병합
             all_videos = []
             for level in ['danger', 'warning', 'safety', 'other']:
                 col_name = f"{level}_videos"
                 if col_name in video_db.list_collection_names(): # 컬렉션이 존재하는지 확인
                      all_videos.extend(list(video_db[col_name].find({'status': 'active'})))
                      total_count += video_db[col_name].count_documents({'status': 'active'})

             # 시간순 정렬 후 페이지네이션 적용
             all_videos.sort(key=lambda x: x.get('created_date', datetime.min), reverse=True)
             videos = all_videos[skip:skip + per_page] # 슬라이싱으로 페이지네이션

        # 응답 데이터 구성
        video_list = []
        for video in videos:
            video_data = {
                "id": str(video["_id"]),
                "risk_level": video.get("risk_level", "unknown"),
                "filename": video.get("filename", "알 수 없음"),
                "created_date": video.get("created_date", datetime.min).strftime("%Y-%m-%d") if isinstance(video.get("created_date"), datetime) else "N/A",
                "created_time": video.get("created_date", datetime.min).strftime("%H:%M:%S") if isinstance(video.get("created_date"), datetime) else "N/A",
                "file_size": video.get("file_size", 0),
                "steal_score": video.get("steal_score"),
                "cctv_name": video.get("cctv_name", "알 수 없음"),
                "duration": video.get("duration"),
                "video_url": url_for('serve_video', file_id=str(video["_id"])),
                "timestamp": video.get('created_date', datetime.min).isoformat() # timestamp 필드 추가
            }
            video_list.append(video_data)

        response = {
            "videos": video_list,
            "pagination": {
                "total_items": total_count,
                "total_pages": (total_count + per_page - 1) // per_page if per_page > 0 else 1,
                "current_page": page,
                "per_page": per_page
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"비디오 필터링 API 오류: {e}")
        return jsonify({"error": "비디오 목록 조회 중 오류가 발생했습니다.", "details": str(e)}), 500

def test_mongodb_connection():
    """MongoDB 연결 상태를 테스트하고 컬렉션 정보를 출력"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['save_to_data_video']
        
        # 데이터베이스 목록 출력
        print("\n=== MongoDB 연결 상태 ===")
        print(f"사용 가능한 데이터베이스: {client.list_database_names()}")
        
        # 컬렉션 정보 출력
        collections = db.list_collection_names()
        print(f"\n현재 데이터베이스의 컬렉션:")
        for collection in collections:
            count = db[collection].count_documents({})
            print(f"- {collection}: {count}개의 문서")
        
        # 각 컬렉션의 최근 문서 확인
        print("\n각 컬렉션의 최근 문서:")
        for collection in collections:
            recent_doc = db[collection].find_one(sort=[('created_date', -1)])
            if recent_doc:
                print(f"\n{collection}의 최근 문서:")
                print(f"- 생성일: {recent_doc.get('created_date', 'N/A')}")
                print(f"- 위험도: {recent_doc.get('risk_level', 'N/A')}")
                print(f"- 파일명: {recent_doc.get('filename', 'N/A')}")
        
        print("\n=== MongoDB 연결 테스트 완료 ===\n")
        return True
        
    except Exception as e:
        print(f"\nMongoDB 연결 테스트 실패: {e}")
        return False

# ========================== 앱 실행 ===========================
if __name__ == '__main__':
    try:
        # MongoDB 연결 테스트
        if not test_mongodb_connection():
            logger.error("MongoDB 연결 테스트 실패")
            raise Exception("MongoDB 연결 실패")
        
            
        scheduler.start()
        logger.info("스케줄러 시작됨 (자동 삭제 및 setup_db)")
        setup_camera_system()
        run_setup_db()
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("스케줄러 종료됨")
