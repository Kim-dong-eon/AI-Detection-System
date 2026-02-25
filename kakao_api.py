import requests
import json

# 1. 발급받은 액세스 토큰
access_token = '79mYJajdkF31t-ufNrvxY-tsrBbnjc7jAAAAAQoNIBsAAAGXmtEEQ_6hmr4nKm-b'

# 2. 요청 URL
url = 'https://kapi.kakao.com/v2/api/talk/memo/default/send'

def send_kakao_message(message: str):
    # 3. 템플릿 메시지 (기본 텍스트 메시지)
    template_object = {
        "object_type": "text",
        "text": message, # 메시지 내용을 인자로 받아 설정
        "link": {
            "web_url": "https://34e4-223-194-255-21.ngrok-free.app/login", # 적절한 URL로 변경
            "mobile_web_url": "https://34e4-223-194-255-21.ngrok-free.app/login" # 적절한 URL로 변경
        },
        "button_title": "CCTV 모니터링"
    }

    # 4. POST 요청
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'template_object': json.dumps(template_object)
    }

    response = requests.post(url, headers=headers, data=data)

    # 5. 결과 확인
    if response.status_code == 200:
        print("✅ 카카오 메시지 전송 성공!")
    else:
        print("❌ 카카오 메시지 전송 실패")
        print(response.status_code, response.text)
