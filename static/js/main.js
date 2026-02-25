// 카메라 설정
const CAMERAS = {
    0: { name: 'CCTV-A002', element: 'cam0', status: 'status0' },
    1: { name: 'CCTV-B003', element: 'cam1', status: 'status1' },
    2: { name: 'CCTV-B001', element: 'cam2', status: 'status2' },
    3: { name: 'CCTV-A003', element: 'cam3', status: 'status3' }
};

let connectedCameras = 0;

// 카메라 상태 관리
let cameraStatuses = {};

// 카메라 스트림 초기화
function initializeCamera(cameraId) {
    const camera = CAMERAS[cameraId];
    const img = document.getElementById(camera.element);
    const statusElement = document.getElementById(camera.status);

    // 카메라 연결 상태 확인
    fetch(`/video_feed/${cameraId}`, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                // 카메라 연결 성공
                img.src = `/video_feed/${cameraId}`;
                statusElement.innerHTML = `${camera.name} <span style="color:green">[연결됨]</span>`;
                connectedCameras++;
                updateCameraCount();
            } else {
                // 카메라 연결 실패
                img.src = '/static/img/disconnected.jpg';
                statusElement.innerHTML = `${camera.name} <span style="color:red">[연결끊김]</span>`;
                updateCameraCount();
            }
        })
        .catch(error => {
            console.error(`카메라 ${camera.name} 연결 확인 중 오류:`, error);
            img.src = '/static/img/disconnected.jpg';
            statusElement.innerHTML = `${camera.name} <span style="color:red">[오류]</span>`;
            updateCameraCount();
        });
}

// 연결된 카메라 수 업데이트
function updateCameraCount() {
    document.getElementById('count-connected').textContent = connectedCameras;
    document.getElementById('count-disconnected').textContent = Object.keys(CAMERAS).length - connectedCameras;
}

// 모든 카메라 초기화
function initializeAllCameras() {
    connectedCameras = 0;
    Object.keys(CAMERAS).forEach(cameraId => {
        initializeCamera(parseInt(cameraId));
    });
}

// 페이지 로드 시 카메라 초기화
document.addEventListener('DOMContentLoaded', () => {
    initializeAllCameras();
    
    // 30초마다 카메라 상태 갱신
    setInterval(initializeAllCameras, 30000);
});

// 카메라 재연결 시도
function retryCamera(cameraId) {
    const camera = CAMERAS[cameraId];
    const statusElement = document.getElementById(camera.status);
    statusElement.innerHTML = `${camera.name} <span style="color:blue">[재연결 중...]</span>`;
    initializeCamera(cameraId);
}

// 에러 처리
window.onerror = function(msg, url, lineNo, columnNo, error) {
    console.error('Error: ' + msg + '\nURL: ' + url + '\nLine: ' + lineNo + '\nColumn: ' + columnNo + '\nError object: ' + JSON.stringify(error));
    return false;
};

// 카메라 상태 업데이트 함수
async function updateCameraStatus() {
    try {
        const response = await fetch('/api/camera-status');
        if (!response.ok) throw new Error('카메라 상태 조회 실패');
        
        const statusData = await response.json();
        cameraStatuses = statusData;
        
        // UI 업데이트
        Object.entries(statusData).forEach(([cameraId, info]) => {
            const statusElement = document.querySelector(`#camera-${cameraId}-status`);
            const reconnectBtn = document.querySelector(`#camera-${cameraId}-reconnect`);
            
            if (statusElement) {
                statusElement.className = `camera-status status-${info.status}`;
                statusElement.textContent = getStatusText(info.status);
            }
            
            if (reconnectBtn) {
                reconnectBtn.style.display = info.status !== 'connected' ? 'block' : 'none';
            }
        });
    } catch (error) {
        console.error('카메라 상태 업데이트 중 오류:', error);
    }
}

// 상태 텍스트 변환
function getStatusText(status) {
    const statusMap = {
        'connected': '연결됨',
        'disconnected': '연결 안됨',
        'error': '오류',
        'failed': '연결 실패'
    };
    return statusMap[status] || '알 수 없음';
}

// 카메라 재연결 함수
async function reconnectCamera(cameraId) {
    try {
        const response = await fetch(`/api/camera-reconnect/${cameraId}`);
        const result = await response.json();
        
        if (response.ok) {
            showNotification('성공', result.message, 'success');
            await updateCameraStatus();
        } else {
            showNotification('오류', result.message, 'error');
        }
    } catch (error) {
        console.error('카메라 재연결 중 오류:', error);
        showNotification('오류', '카메라 재연결 중 오류가 발생했습니다.', 'error');
    }
}

// 알림 표시 함수
function showNotification(title, message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <h4>${title}</h4>
        <p>${message}</p>
    `;
    
    document.body.appendChild(notification);
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// 주기적인 상태 업데이트
setInterval(updateCameraStatus, 30000);

// 페이지 로드 시 초기 상태 업데이트
document.addEventListener('DOMContentLoaded', () => {
    updateCameraStatus();
});
