#!/bin/bash

# LLM 모니터링 및 비교 테스트 스크립트
# vLLM과 Ollama 비교 테스트 및 모니터링 API 검증

BASE_URL="${BASE_URL:-http://localhost:3098}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 LLM 모니터링 및 비교 테스트"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 1. Health Check (vLLM + Ollama)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: 헬스 체크 (vLLM + Ollama)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# 2. vLLM 상태 확인
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: vLLM 상태 확인"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
VLLM_STATUS=$(curl -s "$BASE_URL/health" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data['services']['vllm']['status'])")
echo "vLLM Status: $VLLM_STATUS"

if [ "$VLLM_STATUS" == "healthy" ]; then
    echo "✅ vLLM is running"
else
    echo "⚠️  vLLM is not available ($VLLM_STATUS)"
    echo "   Hint: Start vLLM server first"
    echo "   Command: vllm serve openai/gpt-oss-20b --host 0.0.0.0 --port 8000"
fi
echo ""

# 3. Ollama 상태 확인
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Ollama 상태 확인"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
OLLAMA_STATUS=$(curl -s "$BASE_URL/health" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data['services']['ollama']['status'])")
echo "Ollama Status: $OLLAMA_STATUS"

if [ "$OLLAMA_STATUS" == "healthy" ]; then
    echo "✅ Ollama is running"
else
    echo "⚠️  Ollama is not available ($OLLAMA_STATUS)"
    echo "   Hint: Start Ollama server first"
    echo "   Command: ollama serve"
fi
echo ""

# 4. 모니터링 API: 실시간 메트릭
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: 실시간 메트릭 조회"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$BASE_URL/api/monitoring/llm/metrics/realtime" | python3 -m json.tool
echo ""

# 5. 모니터링 API: LLM 통계 (최근 60분)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: LLM 통계 (최근 60분)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$BASE_URL/api/monitoring/llm/stats?time_window=60" | python3 -m json.tool
echo ""

# 6. 모니터링 API: vLLM vs Ollama 비교
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: vLLM vs Ollama 비교 (최근 60분)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$BASE_URL/api/monitoring/llm/comparison?time_window=60" | python3 -m json.tool
echo ""

# 7. 대시보드 데이터
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: 대시보드 데이터 (통합)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s "$BASE_URL/api/monitoring/dashboard?time_window=60" | python3 -m json.tool
echo ""

# 8. 테스트 질문 전송 (성능 비교용)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: 테스트 질문 전송 (메트릭 생성)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 샘플 질문
QUERY="프로이트의 성격 이론을 간단히 설명해주세요."
ISBN="9791167070982"
SESSION_ID="test_monitoring_$(date +%s)"

echo "질문: $QUERY"
echo "세션: $SESSION_ID"
echo ""

# 질문 전송 (현재 활성 엔진 사용)
echo "📤 질문 전송 중..."
RESPONSE=$(curl -s -X POST "$BASE_URL/api/chat/v1/response" \
  -H "Content-Type: application/json" \
  -d "{
    \"isbn\": \"$ISBN\",
    \"query\": \"$QUERY\",
    \"userid\": \"test_user\",
    \"sessionid\": \"$SESSION_ID\"
  }")

echo "✅ 응답 수신"
echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); result = data.get('result', ''); print(f\"  응답 길이: {len(result)}자\"); print(f\"  응답 미리보기: {result[:50]}...\" if len(result) > 50 else f\"  응답: {result}\")"
echo ""

# 9. 업데이트된 통계 확인
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 9: 업데이트된 통계 확인"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
sleep 1  # 메트릭 수집 대기
curl -s "$BASE_URL/api/monitoring/llm/stats?time_window=5" | python3 -m json.tool
echo ""

# 10. 리포트 내보내기
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 10: 비교 리포트 내보내기"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
curl -s -X POST "$BASE_URL/api/monitoring/llm/export-report" | python3 -m json.tool
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 테스트 완료!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 모니터링 엔드포인트:"
echo "  - 헬스 체크: $BASE_URL/health"
echo "  - 실시간 메트릭: $BASE_URL/api/monitoring/llm/metrics/realtime"
echo "  - LLM 통계: $BASE_URL/api/monitoring/llm/stats"
echo "  - 엔진 비교: $BASE_URL/api/monitoring/llm/comparison"
echo "  - 대시보드: $BASE_URL/api/monitoring/dashboard"
echo "  - Swagger UI: $BASE_URL/docs"
echo ""
