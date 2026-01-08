#!/bin/bash

# 서버 중지 스크립트
# 사용법: ./stop_server.sh

echo "================================"
echo "🛑 PDF AI 챗봇 서버 중지"
echo "================================"
echo ""

# 서버 프로세스 찾기
server_pids=$(ps aux | grep "python -m uvicorn app.main:app" | grep -v grep | awk '{print $2}')

if [ -z "$server_pids" ]; then
    echo "ℹ️  실행 중인 서버가 없습니다."
    exit 0
fi

# 프로세스 중지
echo "🔍 발견된 프로세스:"
ps aux | grep "python -m uvicorn app.main:app" | grep -v grep

echo ""
echo "중지 중..."
for pid in $server_pids; do
    kill $pid
    echo "   PID $pid 중지 요청"
done

sleep 2

# 확인
remaining=$(ps aux | grep "python -m uvicorn app.main:app" | grep -v grep | wc -l)
if [ $remaining -eq 0 ]; then
    echo ""
    echo "✅ 서버 중지 완료"
else
    echo ""
    echo "⚠️  일부 프로세스가 남아있습니다. 강제 종료 중..."
    for pid in $server_pids; do
        kill -9 $pid 2>/dev/null
    done
    echo "✅ 서버 강제 종료 완료"
fi
