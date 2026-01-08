#!/bin/bash

# 서버 시작 스크립트 (포그라운드 실행 - 로그 직접 출력)
# 사용법: ./start_server.sh

echo "================================"
echo "📚 PDF AI 챗봇 서버 시작"
echo "================================"
echo ""
echo "포트: 8080"
echo "로그: 터미널에 직접 출력"
echo "중지: Ctrl+C"
echo ""
echo "================================"
echo ""

# 가상환경 활성화
if [ -d "venv" ]; then
    echo "✅ 가상환경 활성화 중..."
    source venv/bin/activate
else
    echo "❌ 가상환경이 없습니다. 'python -m venv venv'로 생성하세요."
    exit 1
fi

# Python 캐시 삭제
echo "🧹 Python 캐시 삭제 중..."
find . -name "*.pyc" -delete 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo ""
echo "🚀 서버 시작 중..."
echo ""

# 서버 실행 (포그라운드)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
