document.getElementById("loginForm").addEventListener("submit", function (event) {
    event.preventDefault();
  
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    const errorMessage = document.getElementById("error-message");
  
    fetch("/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ username, password })
    })
      .then(response => {
        if (!response.ok) {
          return response.json().then(data => {
            throw new Error(data.message || "로그인 실패");
          });
        }
        return response.json();
      })
      .then(data => {
        if (data.success) {
          window.location.href = "/main"; // 로그인 성공 시 대시보드로 이동
        } else {
          errorMessage.textContent = data.message || "아이디 또는 비밀번호가 올바르지 않습니다.";
        }
      })
      .catch(error => {
        errorMessage.textContent = error.message;
      });
  });