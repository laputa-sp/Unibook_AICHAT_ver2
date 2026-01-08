#!/bin/bash

# 서버 상태 확인 스크립트
# 사용법: ./server_status.sh

echo "================================"
echo "📊 PDF AI 챗봇 서버 상태"
echo "================================"
echo ""

# 1. 프로세스 확인
echo "1️⃣  프로세스 상태:"
echo "---"
server_pids=$(ps aux | grep "python -m uvicorn app.main:app" | grep -v grep)
if [ ! -z "$server_pids" ]; then
    echo "✅ 서버 실행 중"
    echo "$server_pids" | awk '{printf "   PID: %s, CPU: %s%%, MEM: %s%%\n", $2, $3, $4}'
else
    echo "❌ 서버 중지 상태"
fi
echo ""

# 2. 포트 확인
echo "2️⃣  포트 상태:"
echo "---"
if command -v netstat &> /dev/null; then
    port_status=$(netstat -tuln | grep ":3098 " | head -1)
    if [ ! -z "$port_status" ]; then
        echo "✅ 포트 3098 LISTENING"
    else
        echo "❌ 포트 3098 사용 안 함"
    fi
elif command -v ss &> /dev/null; then
    port_status=$(ss -tuln | grep ":3098 " | head -1)
    if [ ! -z "$port_status" ]; then
        echo "✅ 포트 3098 LISTENING"
    else
        echo "❌ 포트 3098 사용 안 함"
    fi
else
    echo "⚠️  netstat/ss 명령어 없음 - 포트 확인 불가"
fi
echo ""

# 3. API 헬스 체크
echo "3️⃣  API 헬스 체크:"
echo "---"
if command -v curl &> /dev/null; then
    health_response=$(curl -s -w "\n%{http_code}" http://localhost:3098/health 2>/dev/null)
    http_code=$(echo "$health_response" | tail -n 1)
    
    if [ "$http_code" == "200" ]; then
        echo "✅ API 정상 응답 (HTTP $http_code)"
        response_body=$(echo "$health_response" | head -n -1)
        echo "   $response_body" | python3 -m json.tool 2>/dev/null | head -5
    else
        echo "❌ API 응답 없음 (HTTP $http_code)"
    fi
else
    echo "⚠️  curl 명령어 없음 - API 확인 불가"
fi
echo ""

# 4. Ollama 상태
echo "4️⃣  Ollama 상태:"
echo "---"
if command -v curl &> /dev/null; then
    ollama_response=$(curl -s http://localhost:11434/api/tags 2>/dev/null)
    if [ ! -z "$ollama_response" ]; then
        echo "✅ Ollama 실행 중"
        model_count=$(echo "$ollama_response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null)
        if [ ! -z "$model_count" ]; then
            echo "   모델 개수: $model_count"
        fi
    else
        echo "❌ Ollama 중지 상태"
    fi
else
    echo "⚠️  curl 명령어 없음 - Ollama 확인 불가"
fi
echo ""

# 5. Qdrant 상태
echo "5️⃣  Qdrant 상태:"
echo "---"
if command -v curl &> /dev/null; then
    qdrant_response=$(curl -s http://localhost:6333/collections 2>/dev/null)
    if [ ! -z "$qdrant_response" ]; then
        echo "✅ Qdrant 실행 중"
        collection_count=$(echo "$qdrant_response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('result', {}).get('collections', [])))" 2>/dev/null)
        if [ ! -z "$collection_count" ]; then
            echo "   컬렉션 개수: $collection_count"
        fi
    else
        echo "❌ Qdrant 중지 상태"
    fi
else
    echo "⚠️  curl 명령어 없음 - Qdrant 확인 불가"
fi
echo ""

# 6. 로그 파일
echo "6️⃣  로그 파일:"
echo "---"
if [ -f "logs/server.log" ]; then
    log_size=$(du -h logs/server.log | awk '{print $1}')
    log_lines=$(wc -l < logs/server.log)
    echo "✅ logs/server.log"
    echo "   크기: $log_size, 줄 수: $log_lines"
elif [ -f "/tmp/backend.log" ]; then
    log_size=$(du -h /tmp/backend.log | awk '{print $1}')
    log_lines=$(wc -l < /tmp/backend.log)
    echo "✅ /tmp/backend.log"
    echo "   크기: $log_size, 줄 수: $log_lines"
else
    echo "⚠️  로그 파일 없음"
fi
echo ""

echo "================================"
echo "💡 유용한 명령어:"
echo "   ./view_logs.sh       로그 보기"
echo "   ./restart_server.sh  서버 재시작"
echo "   ./stop_server.sh     서버 중지"
echo "================================"
