fetch("/api/statistics/cctv")
  .then(res => res.json())
  .then(data => {
    const labels = data.map(d => d._id);        // CCTV 이름 (예: CCTV-1, CCTV-2)
    const counts = data.map(d => d.count);      // 감지 횟수

    new Chart(document.getElementById("cctvChart"), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: '도난 감지 횟수',
          data: counts,
          backgroundColor: 'rgba(255, 99, 132, 0.7)'
        }]
      },
      options: {
        scales: {
          y: { beginAtZero: true }
        }
      }
    });
  })
  .catch(error => {
    console.error("📉 통계 데이터를 불러오는 중 오류 발생:", error);
    alert("CCTV 통계 데이터를 불러오는 데 실패했습니다.");
  });

  fetch("/api/visualizations")
  .then(res => res.json())
  .then(images => {
    const container = document.getElementById("visualizations");
    if (images.length === 0) {
      container.innerHTML = "<p>저장된 추론 이미지가 없습니다.</p>";
      return;
    }
    container.innerHTML = images.map(img =>
      `<img src="${img}" width="320" style="margin:10px; border:1px solid #ccc;">`
    ).join('');
  })
  .catch(error => {
    console.error("📷 추론 이미지 불러오기 실패:", error);
  });
