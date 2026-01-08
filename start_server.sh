#!/bin/bash

# 서버 시작 스크립트 (포그라운드 실행 - 로그 직접 출력)
# 사용법: ./start_server.sh

echo "================================"
echo "📚 PDF AI 챗봇 서버 시작"
echo "================================"
echo ""
echo "포트: 7861 (또는 .env에서 설정한 포트)"
echo "로그: 터미널에 직접 출력"
echo "중지: Ctrl+C"
echo ""
echo "================================"
echo ""

# .env 파일 확인
if [ ! -f ".env" ]; then
    echo "⚠️  .env 파일이 없습니다."
    echo "📝 .env.example을 복사해서 .env 파일을 만드세요:"
    echo "   cp .env.example .env"
    echo ""
    echo "그런 다음 .env 파일에서 다음 설정을 확인하세요:"
    echo "   - VLLM_BASE_URL (vLLM Docker 컨테이너 주소)"
    echo "   - QDRANT_HOST, QDRANT_PORT (Qdrant 설정)"
    echo ""
    exit 1
fi

# 가상환경 활성화
if [ -d "venv" ]; then
    echo "✅ 가상환경 활성화 중..."
    source venv/bin/activate
else
    echo "❌ 가상환경이 없습니다."
    echo "📝 다음 명령으로 가상환경을 생성하고 패키지를 설치하세요:"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    echo ""
    exit 1
fi

# Python 캐시 삭제
echo "🧹 Python 캐시 삭제 중..."
find . -name "*.pyc" -delete 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo ""
echo "🚀 서버 시작 중..."
echo ""

# 서버 실행 (포그라운드) - run.py가 .env에서 포트 읽음
python run.py
