#!/bin/bash

# 서버 시작 스크립트 (백그라운드 실행 + 로그 tail)
# 사용법: ./start_server_bg.sh

echo "================================"
echo "📚 PDF AI 챗봇 서버 시작 (백그라운드)"
echo "================================"
echo ""

# 가상환경 확인
if [ ! -d "venv" ]; then
    echo "❌ 가상환경이 없습니다. 'python -m venv venv'로 생성하세요."
    exit 1
fi

# 기존 서버 중지
echo "🛑 기존 서버 프로세스 확인 중..."
existing_pid=$(ps aux | grep "python -m uvicorn app.main:app" | grep -v grep | awk '{print $2}')
if [ ! -z "$existing_pid" ]; then
    echo "   기존 프로세스 발견 (PID: $existing_pid)"
    echo "   중지 중..."
    kill $existing_pid
    sleep 2
    echo "✅ 기존 서버 중지 완료"
fi

# Python 캐시 삭제
echo "🧹 Python 캐시 삭제 중..."
find . -name "*.pyc" -delete 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 로그 디렉토리 생성
mkdir -p logs

# 서버 시작 (백그라운드)
echo ""
echo "🚀 서버 시작 중..."
source venv/bin/activate
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload > logs/server.log 2>&1 &
server_pid=$!

sleep 2

# 프로세스 확인
if ps -p $server_pid > /dev/null; then
    echo "✅ 서버 시작 완료!"
    echo "   PID: $server_pid"
    echo "   포트: 8080"
    echo "   로그: logs/server.log"
    echo ""
    echo "================================"
    echo "📋 실시간 로그 (Ctrl+C로 로그 보기 중지)"
    echo "================================"
    echo ""
    
    # 로그 tail
    tail -f logs/server.log
else
    echo "❌ 서버 시작 실패"
    echo "   로그 확인: cat logs/server.log"
    exit 1
fi
