# 테스트 가이드

이 문서는 Unibook AI Chat ver2 시스템을 테스트하는 방법을 설명합니다.

## 🎨 웹 UI 테스트 (권장)

프로젝트에 포함된 `vllm_chat.html` 파일을 사용하면 브라우저에서 쉽게 테스트할 수 있습니다.

### 1. 설정

`vllm_chat.html` 파일을 열고 API 엔드포인트를 수정하세요:

#### 수정할 위치

**482줄 근처** - PDF 목록 API:
```javascript
// 변경 전
const response = await fetch('http://localhost:8080/api/pdf/list');

// 변경 후 (본인의 포트로)
const response = await fetch('http://localhost:7861/api/pdf/list');
```

**654줄 근처** - 채팅 스트리밍 API:
```javascript
// 변경 전
const response = await fetch('http://localhost:8080/api/chat/v1/response/stream', {

// 변경 후 (본인의 포트로)
const response = await fetch('http://localhost:7861/api/chat/v1/response/stream', {
```

### 2. 실행

1. 서버가 실행 중인지 확인:
   ```bash
   curl http://localhost:7861/health
   ```

2. 브라우저에서 `vllm_chat.html` 파일 열기:
   ```bash
   # macOS
   open vllm_chat.html

   # Linux
   xdg-open vllm_chat.html

   # Windows
   start vllm_chat.html

   # 또는 브라우저로 직접 드래그 앤 드롭
   ```

3. 웹 페이지에서:
   - 왼쪽 사이드바에서 **도서 선택** (건축시공학 또는 목조건축 개론)
   - 질문 입력 후 **질문하기** 버튼 클릭
   - 실시간 스트리밍 답변 확인

### 3. 주요 기능

#### 📚 도서 선택
- 드롭다운에서 PDF 선택
- 선택 시 새 세션 자동 생성
- ISBN, 페이지 수 정보 표시

#### 💬 채팅
- 실시간 스트리밍 응답
- Markdown 포맷 지원
- 이전 대화 컨텍스트 유지

#### ⏱️ 통계
- 총 질문 수
- 평균 응답 시간
- 마지막 응답 시간
- vLLM 평균 속도

#### 💡 빠른 질문
- 사전 정의된 질문 버튼
- 클릭 시 자동 입력

## 🔧 cURL 테스트

명령줄에서 직접 API를 테스트할 수 있습니다.

### Health Check

```bash
curl http://localhost:7861/health
```

**예상 응답**:
```json
{
  "status": "healthy",
  "database": "connected",
  "ollama": "connected",
  "qdrant": "connected"
}
```

### PDF 목록 조회

```bash
curl http://localhost:7861/api/pdf/list
```

**예상 응답**:
```json
{
  "pdfs": [
    {
      "id": "de1e2807-9d89-4621-80d6-c10113fbe745",
      "isbn": "9788000000001",
      "title": "건축시공학(개정판)",
      "page_count": 50
    },
    {
      "id": "7c172725-ec67-41d9-b468-37e2c6180086",
      "isbn": "9788972955610",
      "title": "목조건축 개론",
      "page_count": 95
    }
  ]
}
```

### 일반 채팅 (비스트리밍)

```bash
curl -X POST http://localhost:7861/api/chat/v1/response \
  -H "Content-Type: application/json" \
  -d '{
    "userid": "test_user",
    "sessionid": "test_session_123",
    "isbn": "9788000000001",
    "query": "1장 내용 요약해줘"
  }'
```

### 스트리밍 채팅

```bash
curl -N -X POST http://localhost:7861/api/chat/v1/response/stream \
  -H "Content-Type: application/json" \
  -d '{
    "userid": "test_user",
    "sessionid": "test_session_123",
    "isbn": "9788000000001",
    "query": "건축시공의 기본 원리는?"
  }'
```

**예상 응답 형식** (SSE):
```
data: {"chunk": "건축시공의", "done": false}
data: {"chunk": " 기본 원리는", "done": false}
data: {"chunk": " 다음과 같습니다...", "done": false}
data: {"chunk": "", "done": true, "metadata": {...}}
```

## 🧪 테스트 시나리오

### 시나리오 1: 기본 Q&A

1. 도서 선택: 건축시공학(개정판)
2. 질문: "1장 내용 요약해줘"
3. 예상: 1장의 주요 내용이 요약되어 반환됨

### 시나리오 2: 연속 대화

1. 질문 1: "건축시공이란?"
2. 질문 2: "좀 더 자세히 설명해줘" (문맥 의존)
3. 예상: 이전 답변을 기반으로 더 상세한 설명 제공

### 시나리오 3: 장 범위 요약

1. 질문: "1-3장 요약해줘"
2. 예상: 1장, 2장, 3장의 요약이 순차적으로 제공됨

### 시나리오 4: 키워드 검색

1. 질문: "콘크리트에 대해 알려줘"
2. 예상: FTS5 검색으로 관련 페이지 찾아서 답변

### 시나리오 5: 시맨틱 검색

1. 질문: "건물을 안전하게 짓는 방법은?"
2. 예상: Qdrant 벡터 검색으로 의미상 관련된 내용 검색 후 답변

## 🐛 문제 해결

### "Connection refused" 에러

**원인**: 서버가 실행되지 않았거나 포트가 다름

**해결**:
```bash
# 서버 상태 확인
./server_status.sh

# 서버 재시작
./restart_server.sh

# 로그 확인
./view_logs.sh
```

### "CORS 에러" (브라우저)

**원인**: 브라우저에서 로컬 HTML 파일을 열 때 CORS 정책 위반

**해결**:
1. Chrome을 CORS 비활성화 모드로 실행:
   ```bash
   # macOS
   open -n -a "Google Chrome" --args --disable-web-security --user-data-dir="/tmp/chrome_dev"

   # Windows
   chrome.exe --disable-web-security --user-data-dir="C:\tmp\chrome_dev"
   ```

2. 또는 간단한 HTTP 서버 실행:
   ```bash
   python -m http.server 8000
   # 브라우저에서 http://localhost:8000/vllm_chat.html 접속
   ```

### "Cannot load PDF list"

**원인**: API 엔드포인트 URL이 잘못됨

**해결**:
1. `vllm_chat.html`의 API URL 확인
2. 서버 포트가 7861인지 확인
3. Health check로 서버 정상 작동 확인

### 느린 응답 시간

**원인**: vLLM 컨테이너가 실행되지 않았거나 GPU 리소스 부족

**해결**:
```bash
# vLLM 컨테이너 상태 확인
docker ps | grep vllm

# vLLM 로그 확인
docker logs vllm_gpt

# GPU 사용량 확인
nvidia-smi
```

## 📊 성능 벤치마크

### 예상 응답 시간 (vLLM, GPU 환경)

| 작업 유형 | 응답 시간 | 설명 |
|----------|---------|------|
| TOC 조회 | 3-5초 | 목차 기반 네비게이션 |
| 페이지 조회 | 2-4초 | 특정 페이지 직접 조회 |
| 키워드 검색 | 4-6초 | FTS5 전문 검색 |
| 시맨틱 검색 | 5-8초 | Qdrant 벡터 검색 |
| 요약 생성 | 8-15초 | 장/절 단위 요약 |
| 연속 대화 | 5-10초 | 대화 컨텍스트 포함 |

### 캐시 효과

- 반복 질문: 99.9% 속도 향상 (20초 → 0.02초)
- 문맥 의존 질문: 캐싱 제외 (점진적 상세화 지원)

## 📝 로그 확인

### 실시간 로그 모니터링

```bash
# 전체 로그
./view_logs.sh

# 또는 tail로 실시간 확인
tail -f logs/app.log
```

### 주요 로그 메시지

**정상 작동**:
```
✅ vLLM health check passed
🎯 Filtered TOC to chapters 1-3 (12 entries, 450 chars)
✅ CACHE HIT for query: 건축시공이란?
```

**오류 상황**:
```
❌ vLLM connection failed, falling back to Ollama
⚠️ TOC truncated to 6000 tokens
❌ JSON extraction failed after all strategies
```

## 🚀 다음 단계

테스트가 성공적으로 완료되었다면:

1. **데이터 추가**: 새로운 PDF 업로드 및 인덱싱
2. **프롬프트 튜닝**: `app/services/llm_service.py`에서 프롬프트 최적화
3. **성능 개선**: 캐시 크기, 토큰 제한 등 조정
4. **프로덕션 배포**: PM2로 프로세스 관리 및 모니터링

---

**테스트 중 문제가 발생하면 이슈를 등록해주세요!**
https://github.com/laputa-sp/Unibook_AICHAT_ver2/issues
