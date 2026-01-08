#!/bin/bash

# 로그 보기 스크립트
# 사용법: ./view_logs.sh [옵션]
#   옵션 없음: 실시간 로그 (tail -f)
#   -n 100: 최근 100줄만 보기
#   -e "ERROR": 에러만 필터링

echo "================================"
echo "📋 서버 로그 보기"
echo "================================"
echo ""

# 로그 파일 확인
if [ -f "logs/server.log" ]; then
    LOG_FILE="logs/server.log"
elif [ -f "/tmp/backend.log" ]; then
    LOG_FILE="/tmp/backend.log"
else
    echo "❌ 로그 파일을 찾을 수 없습니다."
    echo "   예상 위치: logs/server.log 또는 /tmp/backend.log"
    exit 1
fi

echo "📄 로그 파일: $LOG_FILE"
echo ""

# 옵션 처리
if [ "$1" == "-n" ] && [ ! -z "$2" ]; then
    # 마지막 N줄만 보기
    echo "최근 $2줄:"
    echo "================================"
    tail -n $2 $LOG_FILE
elif [ "$1" == "-e" ] && [ ! -z "$2" ]; then
    # 특정 패턴 필터링
    echo "필터: $2"
    echo "================================"
    grep "$2" $LOG_FILE | tail -n 50
elif [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "사용법:"
    echo "  ./view_logs.sh          실시간 로그 (Ctrl+C로 중지)"
    echo "  ./view_logs.sh -n 100   최근 100줄만 보기"
    echo "  ./view_logs.sh -e ERROR 에러만 필터링"
    echo "  ./view_logs.sh -h       도움말"
else
    # 실시간 로그
    echo "실시간 로그 (Ctrl+C로 중지):"
    echo "================================"
    tail -f $LOG_FILE
fi
