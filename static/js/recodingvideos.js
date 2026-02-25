// static/js/video_list.js

function getRiskClass(risk_level) {
    switch(risk_level) {
        case 'danger': return 'risk-danger';
        case 'warning': return 'risk-warning';
        case 'safety': return 'risk-safe';
        default: return '';
    }
}

function getRiskBadge(risk_level) {
    const badge_class = {
        'danger': 'danger',
        'warning': 'warning',
        'safety': 'success'
    }[risk_level] || 'secondary';

    const risk_text = {
        'danger': '위험',
        'warning': '주의',
        'safety': '안전'
    }[risk_level] || '알 수 없음';

    return `<span class="badge bg-${badge_class}">${risk_text}</span>`;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function loadVideos(risk_level = 'all') {
    // 버튼 상태 업데이트
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.filter === risk_level) {
            btn.classList.add('active');
        }
    });

    // 영상 로딩
    fetch(`/api/filtered-videos?risk_level=${risk_level}`)
        .then(response => {
            if (!response.ok) {
                // 서버에서 오류 응답을 보낸 경우
                return response.json().then(error => {
                    throw new Error(error.details || '영상을 불러오는 중 오류가 발생했습니다.');
                });
            }
            return response.json();
        })
        .then(data => { // 응답 구조가 { "videos": [...], "pagination": {...} } 이므로 data 변수 사용
            const videos = data.videos; // 실제 영상 목록 배열은 data.videos에 있습니다.
            const container = document.getElementById('videos-container');
            container.innerHTML = '';

            if (videos.length === 0) {
                container.innerHTML = `
                    <div class="alert alert-info">
                        저장된 영상이 없습니다.
                    </div>
                `;
                return;
            }

            videos.forEach(video => {
                const videoElement = document.createElement('div');
                videoElement.className = 'video-card mb-4';
                videoElement.innerHTML = `
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            ${getRiskBadge(video.risk_level)}
                            <small class="text-muted">${video.created_date} ${video.created_time}</small>
                        </div>
                        <div class="card-body">
                            <video controls width="100%" class="mb-2">
                                <source src="${video.video_url}" type="video/mp4">
                                브라우저가 비디오를 지원하지 않습니다.
                            </video>
                            <div class="d-flex justify-content-between align-items-center">
                                <small class="text-muted">파일 크기: ${formatFileSize(video.file_size)}</small>
                                <a href="${video.video_url}" download class="btn btn-sm btn-outline-primary">
                                    다운로드
                                </a>
                            </div>
                        </div>
                    </div>
                `;

                container.appendChild(videoElement);
            });
            
            // 페이지네이션 정보 처리 (필요하다면 여기에 추가)
            // const pagination = data.pagination;
            // console.log('Pagination:', pagination);

        })
        .catch(error => {
            console.error('Error loading videos:', error);
            const container = document.getElementById('videos-container');
            container.innerHTML = `
                <div class="alert alert-danger">
                    영상을 불러오는 중 오류가 발생했습니다.<br>(${error.message || error})
                </div>
            `;
        });
}

// 페이지 로드 시 전체 영상 목록 불러오기
document.addEventListener('DOMContentLoaded', () => loadVideos('all'));

// 필터 버튼 클릭 이벤트
function filterVideos(risk_level) {
    loadVideos(risk_level);
}

// 30초마다 영상 목록 새로고침
setInterval(loadVideos, 30000);
  