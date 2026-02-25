// 페이지 이동을 위한 함수
function navigateTo(page) {
    window.location.href = page;
}

// '기록된 영상 가기' 버튼 클릭 시 recodingvideos.html로 이동
document.getElementById('navigateButton').addEventListener('click', function() {
    window.location.href = '/recodingvideos';
});
