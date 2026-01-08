# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**E-Book PDF AI 챗봇** - Node.js/Gemini에서 Python/Ollama로 완전 마이그레이션된 시스템입니다. AI 기반 PDF 분석, Q&A, 요약, 키워드 추출, 시맨틱 검색, 문맥 기반 대화 기능을 제공합니다.

**주요 기술 스택:**
- 백엔드: Python 3.8+ FastAPI (Uvicorn ASGI 서버)
- LLM: Ollama (로컬) gpt-oss:20b 모델 (Google Gemini API 대체)
- 데이터베이스: SQLite + FTS5 전문 검색
- 벡터 DB: Qdrant (시맨틱 검색용)
- 임베딩: sentence-transformers (jhgan/ko-sroberta-multitask)
- PDF 처리: PyPDF2, pdfplumber, pdf2image
- 레거시 Node.js 컴포넌트 일부 존재하나 Python 서비스가 주력

## 핵심 아키텍처 패턴

### 이중 서비스 아키텍처

코드베이스에 **Node.js와 Python** 구현이 공존합니다:
- **Python FastAPI** (app/): 프로덕션 시스템, 포트 7861 (또는 PM2에서 3098)
- **Node.js Express** (레거시 파일): 참조용 구현, 대부분 deprecated

**변경 작업 시 Node.js 호환성을 명시적으로 다루는 경우가 아니라면 Python 코드베이스(`app/` 디렉토리)에서 작업하세요.**

### LLM 서비스 아키텍처 (vLLM + Ollama Dual Engine)

LLM 서비스는 **vLLM과 Ollama를 모두 지원**하며 자동 Fallback 메커니즘을 제공합니다:

**서비스 레이어:**
- `llm_service.py`: 메인 LLM 서비스 (get_answer, get_summary, get_search_type 등)
  - vLLM 우선 시도 → 실패 시 Ollama로 자동 Fallback
  - 원본 Gemini 프롬프트 유지
- `qa_service.py`: QA 흐름 오케스트레이션
- `ollama_service.py`: Ollama 클라이언트 래퍼
- `qdrant_service.py`: 벡터 검색
- `embedding_service.py`: 임베딩 생성
- `chunk_service.py`: 텍스트 청킹

**LLM 엔진 설정** (`.env`):
```bash
LLM_ENGINE=vllm                  # 기본 엔진 (vllm 또는 ollama)
ENABLE_LLM_FALLBACK=true         # Fallback 활성화
FALLBACK_ENGINE=ollama           # Fallback 대상

# vLLM (Docker 컨테이너)
VLLM_BASE_URL=http://vllm_gpt:8000
VLLM_MODEL_NAME=openai/gpt-oss-20b

# Ollama (로컬)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b
```

**Fallback 동작**:
1. vLLM 시도 (~1초 타임아웃)
2. 실패 시 Ollama로 자동 전환
3. 메트릭 수집 (`llm_metrics.py`)

**중요:**
- vLLM 컨테이너는 별도 관리 - 서버 재시작 시 Docker 건드리지 않음
- Ollama 스트리밍: `ChatResponse` 객체 반환 (딕셔너리 아님!)
- Reasoning 모델(gpt-oss:20b): `thinking` → `content` 두 단계 출력

### 대화 컨텍스트 메모리 처리 (중요!)

**문제 번호 참조 패턴**: 사용자가 "문제 1번", "문제 2번" 등을 언급하면 LLM이 대화 기록에서 해당 문제를 찾아 정답을 제공해야 합니다.

프롬프트 위치: `llm_service.py` 453-478줄
```python
**🚨🚨🚨 CRITICAL - 문제 번호 참조는 반드시 대화 기록에서 찾아라! 🚨🚨🚨**

사용자가 "문제 N번" (N=1,2,3...) 을 언급하면:
→ 반드시 아래 대화 기록의 answer에서 해당 번호 문제를 찾아라!
→ 책 내용에서 찾지 마라! 대화 기록에서 찾아라!

**처리 순서:**
STEP 1: 대화 기록 answer에서 "1)", "2)", "문제 1", "문제 2" 텍스트를 직접 찾기
STEP 2: 찾은 문제의 질문 내용 전체를 복사하기
STEP 3: 그 질문에 대한 정답 제공하기
```

**대화 컨텍스트 관련 수정 시 주의사항:**
- 대화 기록은 JSON 형식으로 `recentConversations` (최근 10개)와 `previousQuestions` (이전 질문만)로 구성
- `qa_service.py:_get_conversations_by_session_with_context()` 함수가 대화 기록 포맷팅 담당
- 문제 번호 참조가 작동하지 않으면 프롬프트의 CRITICAL 섹션을 강화하세요

### 한국어 출력 강제

**모든 LLM 응답은 한국어로만 출력되어야 합니다.** 영어, 일본어, 중국어 등 다른 언어 사용 금지.

프롬프트 위치: `llm_service.py` 294줄, 492줄
```python
# 핵심 원칙 1번
1. 🇰🇷 **모든 답변은 반드시 한국어로만 작성** (영어/일본어/중국어 등 절대 금지!)

# 출력 규칙 첫 번째
- 🇰🇷 **반드시 한국어로만 답변** (영어, 일본어, 중국어 등 다른 언어 사용 금지!)
```

**문제 출제 시 영어로 나오는 경우 이 규칙이 적용되지 않은 것이니 프롬프트를 확인하세요.**

### QA 흐름 파이프라인

QA 시스템은 다음 흐름을 따릅니다 (`qa_service.py:handle_json_results()`에 정의):

1. **PDF 메타데이터 가져오기** - 데이터베이스에서 ISBN으로 조회
2. **대화 히스토리 로드** - 캐시에서 userid + sessionid로 조회
3. **`get_search_type()` 호출** - 전략 결정 (toc/page/summary/keyword/semantic/quiz/followup)
4. **searchType에 따라 관련 데이터 수집:**
   - `toc`: 목차 네비게이션
   - `page`: 전문 검색으로 페이지 찾기
   - `summary`: 챕터/섹션 요약
   - `keyword`: 키워드 기반 FTS5 검색
   - `semantic`: Qdrant 벡터 검색 (의미 기반)
   - `quiz`: 퀴즈 생성
   - `followup`: 이전 대화 컨텍스트 사용
5. **수집된 컨텍스트로 `get_answer()` 호출**
6. **대화 히스토리에 결과 캐시**
7. **포맷된 응답 반환**

**QA 이슈 디버깅 시 3번 단계의 searchType 결정부터 확인하세요.**

### 시맨틱 검색 아키텍처 (Qdrant)

**Qdrant 벡터 DB**를 사용한 시맨틱 검색이 FTS5 키워드 검색을 보완합니다:

**청킹 전략** (`chunk_service.py`):
- 고정 크기: 500 토큰 청크, 100 토큰 오버랩
- 페이지 경계 보존: 청크 메타데이터에 page_number 포함
- 토큰 카운팅: tiktoken (cl100k_base 인코더)

**임베딩** (`embedding_service.py`):
- 모델: `jhgan/ko-sroberta-multitask` (한국어 최적화)
- 차원: 768
- 배치 처리: 32개 청크씩

**벡터 검색** (`qdrant_service.py`):
- 컬렉션명: `pdf_{pdf_id}`
- 유사도 메트릭: Cosine
- 기본 검색 제한: 5개 청크
- 메타데이터 필터링: page_number, chunk_index

**검색 흐름:**
1. 사용자 쿼리 → 임베딩 생성
2. Qdrant 벡터 검색 (상위 5개 청크)
3. 검색 결과 → LLM 컨텍스트로 제공
4. 출처 페이지 번호 자동 추출

**Qdrant 디버깅:**
```bash
# Qdrant 서비스 상태 확인
curl http://localhost:6333/collections

# 특정 컬렉션 정보
curl http://localhost:6333/collections/pdf_123

# 벡터 검색 테스트
python -c "from app.services.qdrant_service import qdrant_service; import asyncio; asyncio.run(qdrant_service.search('pdf_id', 'query', limit=5))"
```

### 출처 형식 통일

**모든 출처는 다음 형식을 따라야 합니다:**

```
목차제목 (페이지)
```

**예시:**
- ✅ `13-4 자기강화와 자기효능감 (595-600페이지)`
- ✅ `2-5-2 승자의 미덕 (206-208페이지)`
- ❌ `문화절대주의와 문화상대주의, 1` (쉼표 사용 금지)
- ❌ `13-4 자기강화와 자기효능감 595-600` (괄호 생략 금지)

프롬프트 위치: `llm_service.py` 321-344줄

**출처 병합 규칙:**
- 연속된 단일 페이지(603, 604, 605) → 범위(603-605)로 병합
- 이미 범위로 제공된 출처(8-14, 8-15) → 절대 병합 금지!

### 스트리밍 API

**SSE (Server-Sent Events)** 스트리밍 지원:

엔드포인트: `POST /api/chat/v1/response/stream`

응답 형식:
```
data: {"chunk": "텍스트 조각", "done": false}
data: {"chunk": "", "done": true, "metadata": {...}}
```

구현 위치: `app/routes/chat.py:legacy_api_response_stream()`

**주의사항:**
- 스트리밍 응답은 LLM 캐시를 우회합니다
- `qa_service.py:handle_json_results_stream()` 사용
- 청크 단위로 응답 전송 (문장 단위 권장)

### 데이터베이스 스키마

SQLite 데이터베이스 위치: `uploads/app.db`

주요 테이블:

- `pdfs`: PDF 메타데이터 (id, isbn, title, file_path, page_count, file_size, has_toc)
- `pdf_pages`: 페이지 내용 (id, pdf_id, page_number, content, word_count)
- `pdf_pages_fts`: 페이지 내용 전문 검색용 FTS5 가상 테이블
- `toc_entries`: 목차 정보 (pdf_id, level, title, start_page, end_page, core_summary, summary)
- `visual_elements`: 이미지/수식/표 메타데이터 (id, pdf_id, page_number, type, path, bbox, description)

**중요:** 페이지 내용 검색 시 `LIKE`가 아닌 FTS5 `MATCH` 쿼리를 사용하세요. `pdf_database.py:search_pages()` 참고.

### Visual Elements 통합 (2026-01-08 추가)

**시스템 아키텍처:**
- **Node.js Vision 파이프라인**: PDF → 이미지 변환 (pdftocairo) → Gemini Vision API 호출
- **Python QA Service**: 페이지 검색 시 visual_elements 자동 조회 및 응답에 포함
- **DB 스키마**: `visual_elements` 테이블에 type (image/formula/table), description, bbox 저장

**데이터 흐름:**
1. PDF 업로드 → pdftocairo로 페이지별 PNG 생성
2. Gemini Vision API로 이미지/표/수식 감지 및 텍스트 추출
3. `visual_elements` 테이블에 저장
4. QA 시 페이지 검색되면 해당 페이지의 visual_elements 자동 조회
5. API 응답에 `visualElements` 필드로 반환

**핵심 파일:**
- `app/services/pdf_database.py:get_visual_elements()` (Lines 802-849)
- `app/services/qa_service.py` - 페이지 검색 후 visual_elements 조회 (Lines 495-514, 1232-1255)
- Node.js 스크립트: `test_vision_extraction.js`, `test_vision_architecture.js`

**중요 노트:**
- Visual elements는 Node.js Gemini Vision API로만 추출 가능 (Python은 QA만 담당)
- description만 DB 저장, 실제 이미지 파일은 임시 저장 후 삭제
- 할루시네이션 방지: DB에 존재하는 visual_elements만 반환
- 상세 내용은 `VISUAL_ELEMENTS_TEST_REPORT.md` 참고

### 대화 컨텍스트 관리 (2026-01-05 최신 업데이트)

**3단계 대화 메모리 시스템:**

1. **대화 요약 시스템** (`qa_service.py`):
   - **목적**: 프롬프트 크기 폭발 방지 - 대화가 길어져도 일정한 응답 시간 유지
   - **구조**: `conversation_summaries = {session_id: {"summary": str, "recent_turns": List[Dict]}}`
   - **요약 생성**: LLM이 전체 대화를 3-5문장으로 압축 (백그라운드 실행)
   - **최근 대화**: 최근 5턴만 상세 보관 (Deque 패턴)
   - **프롬프트 포맷**: `[대화 요약]\n{요약}\n\n[최근 대화]\n{최근 3턴}`
   - **핵심 파일**: `qa_service.py:_update_conversation_summary()` (Lines 1038-1069)

2. **스마트 캐싱 시스템** (문맥 의존성 감지):
   - **전체 요청 캐싱**: get_search_type + DB + LLM 전체를 캐싱 → 반복 질문 99.9% 속도 향상 (20초 → 0.02초)
   - **문맥 의존 키워드 검출** (30+ 키워드):
     - 상세도: 더, 자세, 상세, 구체
     - 연속성: 계속, 이어, 추가, 다시
     - 접속사: 그럼, 그러면, 그래서, 하지만
     - 지시어: 이거, 저거, 위, 아래, 이전, 방금
     - 짧은 질문: 네, 응, 왜, 예를 (≤3자 자동 감지)
   - **중요**: 문맥 의존 질문("좀 더 자세하게")은 캐싱 제외 → 점진적 상세화 지원
   - **핵심 파일**: `qa_service.py:handle_json_results()` (Lines 95-115)

3. **LLMCache** (`utils/llm_cache.py`):
   - 크기: 3000개 (기존 1000 → 3배 증가)
   - TTL: 72시간 (기존 24시간 → 3배 증가)
   - 쿼리 정규화: 소문자, 구두점 제거, 동의어 통일
   - 캐시 키: `md5(normalized_query + isbn + search_type)`

**대화 로깅**: JSON 파일로 저장 (`uploads/conversations/YYYY-MM-DD/session_*.json`)
- 세션별 대화 기록
- 메타데이터 포함 (검색 타입, 페이지, 응답 길이)

**후속 질문은 대화 캐시에 의존합니다 - 요청 간 sessionid가 유지되어야 합니다.**

**성능 지표 (2026-01-05 실측):**
- 독립적 질문 반복: 20초 → 0.02초 (99.9% 개선)
- 연속 대화: 13→37초 증가 문제 해결 → 18-20초로 안정화
- 문맥 의존 질문: 점진적 상세화 (247자 → 967자 → 1,669자)

### LLM 모니터링 및 비교 인프라 (2026-01-06 추가)

**vLLM vs Ollama 성능 비교 시스템**을 위한 모니터링 인프라가 구축되었습니다:

**주요 컴포넌트:**
- `app/utils/llm_comparison.py`: 메트릭 수집 및 비교 분석
- `app/routes/monitoring.py`: 모니터링 API 엔드포인트
- `app/routes/health.py`: vLLM/Ollama 헬스 체크
- `test_llm_monitoring.sh`: 통합 테스트 스크립트

**모니터링 API 엔드포인트:**
```bash
GET /health                              # vLLM + Ollama 헬스 체크
GET /api/monitoring/llm/stats            # LLM 통계 (엔진별)
GET /api/monitoring/llm/comparison       # vLLM vs Ollama 비교
GET /api/monitoring/llm/metrics/realtime # 실시간 메트릭
GET /api/monitoring/dashboard            # 통합 대시보드 데이터
POST /api/monitoring/llm/export-report   # 비교 리포트 내보내기
```

**수집 메트릭:**
- Latency: mean, median, P95, P99, min, max
- Throughput: tokens/second, requests/minute
- Token usage: prompt, completion, total
- Success/failure rates
- Cache hit rates

**테스트 실행:**
```bash
./test_llm_monitoring.sh  # 10단계 자동화 테스트
```

**메트릭 로그 위치:**
- `logs/llm_comparison/comparison_YYYY-MM-DD.jsonl` - 실시간 메트릭
- `logs/llm_comparison/comparison_report_*.json` - 비교 리포트

**중요 노트:**
- 메트릭 수집은 자동이지만 아직 `llm_service.py`와 완전 통합되지 않음
- vLLM 컨테이너 시작 후 실제 비교 테스트 가능

## 주요 개발 명령어

### 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt

# Ollama 모델 다운로드
ollama pull gpt-oss:20b

# Qdrant 서비스 시작 (Docker)
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Ollama 확인
ollama list
```

### 서버 실행

**편리한 스크립트 사용 (권장):**

```bash
# 서버 시작 (포그라운드 - 개발 시)
./start_server.sh

# 서버 시작 (백그라운드 - 프로덕션)
./start_server_bg.sh

# 서버 상태 확인
./server_status.sh

# 서버 재시작
./restart_server.sh

# 서버 중지
./stop_server.sh

# 로그 확인
./view_logs.sh
```

**직접 실행 (수동):**

```bash
# 개발 모드 (자동 리로드)
python run.py

# 또는 uvicorn 직접 실행
uvicorn app.main:app --reload --port 3098

# 백그라운드 실행
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 3098 --reload > /tmp/backend.log 2>&1 &

# PM2로 프로덕션 실행
pm2 start ecosystem.config.js
pm2 logs ebook-python-api
pm2 restart ebook-python-api
```

**포트 정보:**
- 개발 기본 포트: 7861
- 프로덕션 포트: 3098 (PM2 및 스크립트 기본값)

### 테스트

```bash
# LLM 서비스 테스트 (6개 테스트)
python test_llm_service.py

# Chat API 통합 테스트 (4개 테스트)
python test_chat_api_integration.py

# Legacy API 호환성 테스트
python test_legacy_api.py

# Qdrant 시맨틱 검색 테스트
python test_qdrant_search.py

# 헬스 체크
curl http://localhost:7861/health

# API 문서 (Swagger UI)
# 브라우저: http://localhost:7861/docs
```

### 데이터베이스 작업

```bash
# SQLite CLI 열기
sqlite3 uploads/app.db

# 테이블 보기
.tables

# PDF 개수 확인
SELECT COUNT(*) FROM pdfs;

# 목차 정보 확인
SELECT * FROM toc_entries WHERE pdf_id='xxx' LIMIT 10;

# FTS5 검색 인덱스 확인
SELECT * FROM pdf_pages_fts WHERE content MATCH '검색어' LIMIT 5;

# 데이터베이스 백업
cp uploads/app.db uploads/app.db.backup
```

### Ollama 작업

```bash
# 설치된 모델 목록
ollama list

# 모델 직접 테스트
ollama run gpt-oss:20b "안녕하세요"

# Ollama 서비스 상태 확인
curl http://localhost:11434/api/tags

# Ollama 재시작 (필요시)
ollama serve
```

### Qdrant 작업

```bash
# 컬렉션 목록 확인
curl http://localhost:6333/collections

# 특정 PDF의 벡터 개수
curl http://localhost:6333/collections/pdf_123

# 컬렉션 삭제 (재인덱싱 필요시)
curl -X DELETE http://localhost:6333/collections/pdf_123
```

## API 엔드포인트 구조

**Health API**
- `GET /health` - 전체 헬스 체크 (DB + Ollama + Qdrant)
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe

**PDF API** (`/api/pdf/`)
- `POST /upload` - PDF 업로드 (multipart form, isbn + title)
- `GET /list` - 모든 PDF 목록
- `GET /isbn/{isbn}` - ISBN으로 PDF 정보 조회
- `GET /{pdf_id}/pages` - 페이지 내용 가져오기
- `GET /{pdf_id}/search?q=query` - FTS5 검색
- `DELETE /{isbn}` - PDF 삭제

**Chat API** (`/api/chat/`)
- `GET /models` - Ollama 모델 목록
- `POST /chat` - 일반 채팅 (선택적 PDF 컨텍스트)
- `POST /ask` - PDF Q&A
- `POST /summarize` - PDF 요약
- `POST /keywords` - 키워드 추출
- `POST /v1/response` - **Legacy API** (Node.js 호환)
- `POST /v1/response/stream` - **Streaming API** (SSE)
- `DELETE /conversation/{id}` - 대화 삭제
- `GET /conversation/stats` - 대화 캐시 통계
- `GET /cache/stats` - LLM 캐시 통계

**Legacy API는 UniBook 앱 통합에 중요 - 정확한 요청/응답 포맷을 보존하세요.**

## 설정

`.env` 환경 변수:

```bash
# 서버
DEBUG=true
HOST=0.0.0.0
PORT=7861

# 데이터베이스
DATABASE_URL=uploads/app.db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# 임베딩
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
EMBEDDING_DEVICE=cpu  # 또는 cuda

# 청킹
CHUNK_SIZE=500  # 토큰 단위
CHUNK_OVERLAP=100  # 토큰 단위

# 파일 업로드
MAX_UPLOAD_SIZE=52428800  # 50MB
```

**포트 7861**은 기본 개발 포트입니다. 프로덕션은 **3098** 사용 (`ecosystem.config.js`에 설정).

## 코드 구성 원칙

### 서비스 레이어 책임

- `llm_service.py`: 무상태 LLM 함수 호출 - 순수 입출력, 비즈니스 로직 없음
- `qa_service.py`: 상태 있는 QA 오케스트레이션 - 대화 컨텍스트, 캐싱, 다단계 흐름 관리
- `pdf_database.py`: 모든 데이터베이스 작업 - async aiosqlite 사용
- `pdf_parser.py`: PDF 파싱 및 이미지 추출
- `qdrant_service.py`: 벡터 DB 작업 - 인덱싱, 검색
- `embedding_service.py`: 텍스트 임베딩 생성
- `chunk_service.py`: 텍스트 청킹 및 토큰 관리

### 라우트 레이어 패턴

라우트(`app/routes/`)는 얇은 컨트롤러여야 합니다:
- 요청 검증 (Pydantic 모델)
- 서비스 레이어 호출
- 응답 포맷
- 에러 핸들링

**비즈니스 로직을 라우트에 넣지 말고 서비스에 유지하세요.**

### 에러 핸들링

FastAPI 예외 핸들러는 `app/main.py`에 있습니다:
- 404: Not Found
- 500: Internal Server Error

서비스 레이어는 설명적인 예외를 발생시키고, 라우트에서 HTTP 응답으로 변환합니다.

## 일반 작업별 주요 파일

**LLM 프롬프트나 동작 수정:**
- `app/services/llm_service.py` - 프롬프트 템플릿 수정 (`_build_answer_prompt` 메서드)
- 핵심 원칙 (294줄), 출력 규칙 (492줄), 대화 컨텍스트 (453줄) 섹션 확인
- **TOC 필터링 최적화** (Lines 1277-1318):
  - **대화 참조 질문** ("1번 문제", "정답", "해설" 등) → TOC 제거하여 토큰 절약
  - **장 번호 질문** ("13-15장") → 해당 장만 필터링
  - 토큰 제한(6000) 초과 시 자동 truncate

**QA 흐름이나 검색 전략 변경:**
- `app/services/qa_service.py` - `handle_json_results()` 또는 `_determine_search_strategy()` 수정

**시맨틱 검색 개선:**
- `app/services/qdrant_service.py` - 벡터 검색 로직
- `app/services/embedding_service.py` - 임베딩 모델 또는 배치 처리
- `app/services/chunk_service.py` - 청킹 전략 (크기, 오버랩)

**새 API 엔드포인트 추가:**
- `app/routes/`에 라우트 생성
- `app/models/`에 Pydantic 모델 정의
- `app/main.py`에 라우터 포함

**데이터베이스 스키마 변경:**
- `app/services/pdf_database.py:init_database()` 수정
- 필요시 마이그레이션 스크립트 생성

**PDF 파싱 개선:**
- `app/services/pdf_parser.py:parse_pdf()`

**대화 컨텍스트 메모리 개선:**
- `app/services/qa_service.py:_update_conversation_summary()` - 대화 요약 생성 (Lines 1038-1069)
- `app/services/qa_service.py:_format_conversation_context()` - 프롬프트용 포맷팅
- `app/services/llm_service.py:_build_answer_prompt()` - 프롬프트 내 대화 컨텍스트 처리 (453-478줄)

**캐싱 및 문맥 의존성 디버깅:**
- `app/services/qa_service.py:handle_json_results()` - 전체 요청 캐싱 + 문맥 의존 키워드 검출 (Lines 95-115)
- `app/services/llm_service.py:get_answer()` - LLM 답변 캐싱 + 문맥 의존 검출 (Lines 317-369)
- `app/utils/llm_cache.py:_get_cache_key()` - 쿼리 정규화 로직 (Lines 41-77)

## 성능 고려사항

### LLM 응답 시간

- TOC/Page 쿼리: ~5-10초
- 요약 생성: ~10-20초
- 키워드 추출: ~5-10초
- Semantic 검색: ~3-8초 (임베딩 + 벡터 검색)
- Followup (컨텍스트 포함): ~5-15초

**타임아웃:** API 기본 타임아웃 60초.

### 캐싱 (2026-01-05 개선)

**4단계 스마트 캐싱 시스템:**

1. **전체 요청 캐싱** (신규 - 가장 빠름):
   - get_search_type + DB 조회 + LLM 생성 전체를 캐싱
   - 효과: 반복 질문 99.9% 속도 향상 (20초 → 0.02초)
   - 제외: 문맥 의존 질문 (30+ 키워드 검출)
   - 캐시 키: `f"full:{isbn}:{query}"`

2. **LLM 답변 캐싱**:
   - 크기: 3000개 (1000 → 3배 증가)
   - TTL: 72시간 (24시간 → 3배 증가)
   - 쿼리 정규화: 소문자, 구두점 제거, 동의어 통일
   - 문맥 의존 질문 제외

3. **대화 요약 캐싱**:
   - 전체 대화를 3-5문장 요약으로 압축
   - 최근 5턴만 상세 보관
   - 프롬프트 크기 일정 유지 → 응답 시간 안정화

4. **임베딩 캐시**:
   - 동일 텍스트 임베딩 재사용
   - LRU 방식 제거

**캐시 무효화:**
- 전체 요청 캐시: 서버 재시작 시
- LLM 캐시: 72시간 TTL 또는 서버 재시작 시
- 대화 캐시: 수동 삭제 (`DELETE /api/chat/conversation/{id}`)
- 임베딩 캐시: 메모리 부족 시 LRU 방식

**캐시 통계 확인:**
```bash
# LLM 캐시 통계
curl http://localhost:3098/api/chat/cache/stats

# 대화 캐시 통계
curl http://localhost:3098/api/chat/conversation/stats
```

### 데이터베이스 성능

- FTS5 검색은 키워드 쿼리에 빠름 (밀리초 단위)
- Qdrant 벡터 검색은 의미 기반 쿼리에 효과적 (100ms 이내)
- 대용량 PDF (500+ 페이지): 페이지 검색 결과에 페이지네이션 사용
- 요약 생성은 전체 책이 아닌 챕터 단위로 제한
- 자주 접근하는 페이지 인덱싱

### 벡터 인덱싱 시간

PDF 업로드 후 벡터 인덱싱은 백그라운드에서 실행:
- 100페이지: ~30-60초
- 500페이지: ~2-5분
- 1000페이지: ~5-10분

인덱싱 상태는 `pdfs.indexed_at` 컬럼으로 확인.

## 마이그레이션 노트 (Gemini → Ollama)

이 시스템은 Gemini에서 Ollama로 완전히 마이그레이션되었습니다. 주요 차이점:

**Gemini** (원본):
- 구조화된 출력 지원하는 클라우드 API
- 빠르고 안정적인 JSON 모드
- 쉬운 함수 호출

**Ollama** (현재):
- 로컬 추론 (느리지만 프라이빗)
- JSON 추출에 커스텀 파싱 필요 (`_extract_json_from_text()`)
- 가끔 포맷 불일치 - `_clean_llm_response()`로 정리

**LLM 이슈 디버깅 시:**
1. 로그에서 원시 Ollama 응답 확인
2. `_extract_json_from_text()`에서 JSON 추출 검증
3. 프롬프트 직접 테스트: `ollama run gpt-oss:20b "your prompt"`
4. 코드 주석의 원본 Gemini 프롬프트와 비교

## PM2 프로덕션 배포

PM2로 프로덕션 서버 관리:

```bash
# 서버 시작
pm2 start ecosystem.config.js

# 상태 확인
pm2 status
pm2 logs ebook-python-api

# 재시작
pm2 restart ebook-python-api

# 중지
pm2 stop ebook-python-api

# 부팅 시 자동 시작 활성화
pm2 startup
pm2 save
```

**PM2 설정** (`ecosystem.config.js`):
- 스크립트: `./venv/bin/python run.py`
- 포트: 3098
- 최대 메모리: 1GB (초과 시 자동 재시작)
- 로그: `logs/out.log`, `logs/err.log`

## 문제 해결

**Ollama 연결 오류:**
- 확인: `curl http://localhost:11434/api/tags`
- 재시작: `ollama serve`

**Qdrant 연결 오류:**
- 확인: `curl http://localhost:6333/collections`
- Docker 재시작: `docker restart qdrant` (컨테이너명 확인 필요)

**데이터베이스 잠금 오류:**
- 서버 인스턴스가 하나만 실행 중인지 확인
- 고아 프로세스 확인: `lsof uploads/app.db`

**JSON 추출 실패:**
- 디버그 로깅 활성화: `.env`에 `DEBUG=true`
- 로그에서 Ollama 응답 포맷 확인
- `_extract_json_from_text()` regex 패턴 조정 필요할 수 있음

**느린 LLM 응답:**
- 모델 다운로드 확인: `ollama list`
- 시스템 리소스 확인: `top` 또는 `htop`
- 테스트용으로 더 작고 빠른 모델 고려

**벡터 검색이 작동하지 않음:**
- PDF가 인덱싱되었는지 확인: `SELECT indexed_at FROM pdfs WHERE id='xxx'`
- Qdrant 컬렉션 존재 확인: `curl http://localhost:6333/collections/pdf_xxx`
- 임베딩 모델 로드 확인: 서버 로그에서 "Embedding model loaded" 메시지

**Followup 타입 테스트 실패:**
- 대화 캐시 작동 확인: `ConversationCache` 로그 체크
- 컨텍스트 + followup 요청 간 sessionid 일관성 확인
- `qa_service.py`에서 이전 컨텍스트가 검색되는지 확인

**문제 번호 참조가 작동하지 않음:**
- 대화 기록이 제대로 저장되는지 확인: `uploads/conversations/` 디렉토리
- `llm_service.py` 453-478줄 프롬프트의 CRITICAL 섹션 확인
- LLM 응답 로그에서 대화 기록이 프롬프트에 포함되었는지 확인

**영어로 응답이 나옴:**
- `llm_service.py` 294줄, 492줄의 한국어 출력 규칙 확인
- Ollama 모델이 한국어를 지원하는지 확인
- 프롬프트가 제대로 전달되었는지 로그 확인

**문맥 의존 질문이 캐싱됨 (같은 답변 반복):**
- `qa_service.py` Lines 95-115의 문맥 의존 키워드 목록 확인
- 로그에서 "Context-dependent query" 메시지 확인
- 새로운 문맥 의존 패턴 발견 시 키워드 목록에 추가
- 매우 짧은 질문(≤3자)은 자동으로 문맥 의존으로 처리됨

**대화가 길어지면 응답 시간이 증가함:**
- 대화 요약 시스템이 작동하는지 확인: 로그에서 "Generating conversation summary" 확인
- `conversation_summaries` 딕셔너리가 제대로 업데이트되는지 확인
- 백그라운드 실행 문제 발생 시 `asyncio.create_task()` 체크
- 최대 턴 수(MAX_RECENT_TURNS=5) 확인

**캐시가 작동하지 않음 (반복 질문이 느림):**
- 캐시 통계 확인: `curl http://localhost:3098/api/chat/cache/stats`
- 로그에서 "CACHE HIT" 또는 "FULL CACHE HIT" 메시지 확인
- 서버 재시작으로 메모리 캐시가 초기화됨
- 문맥 의존 키워드가 포함된 질문은 의도적으로 캐싱 안 됨

**점진적 상세화가 작동하지 않음 ("좀 더 자세하게"가 항상 같은 길이):**
- 문맥 의존 키워드 검출이 작동하는지 확인
- 대화 요약이 프롬프트에 포함되는지 확인
- 로그에서 "Context-dependent, skipping cache" 메시지 확인
- LLM에 전달되는 프롬프트에 이전 답변 내용이 포함되는지 확인

## 문서

`app/docs/`의 전체 문서:
- 00_최종_점검_리포트.md
- 01_프로젝트_개요.md
- 02_개발환경_세팅.md
- 03_API_구현.md
- 04_LLM_서비스_마이그레이션.md
- 05_데이터베이스_구조.md
- 06_Chat_API_LLM_통합.md
- 07_프로덕션_환경_통합.md
- 08_프로덕션_테스트_결과.md

루트 디렉토리의 주요 문서:
- `QDRANT_MIGRATION_ANALYSIS.md` - 시맨틱 검색 마이그레이션 분석
- `QDRANT_TEST_RESULTS.md` - Qdrant 통합 테스트 결과
- `CACHE_STREAMING_IMPLEMENTATION.md` - 캐싱 및 스트리밍 구현
- `PERFORMANCE_OPTIMIZATION.md` - 성능 최적화 가이드

작업 일지 (최신순):
- `WORK_LOG_20260105.md` - 대화 요약 시스템 + 스마트 캐싱 (문맥 의존성 감지)
- `WORK_LOG_20260102.md` - Qdrant 시맨틱 검색 통합
- `WORK_LOG_20251222.md` - 캐시 및 스트리밍 구현
- `WORK_LOG_20251231.md` - UI/UX 개선

**프로젝트 배경은 01_프로젝트_개요.md, API 사용법은 03_API_구현.md부터 시작하세요.**

**최신 성능 개선 및 캐싱 시스템은 WORK_LOG_20260105.md를 참조하세요.**
