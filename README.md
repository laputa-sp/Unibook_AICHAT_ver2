# Unibook AI Chat ver2

**E-Book PDF AI 챗봇** - Python/FastAPI 기반 AI 학습 도우미 시스템

## 📚 프로젝트 개요

PDF 교재를 AI와 대화하며 학습할 수 있는 시스템입니다. 질문 답변, 요약, 키워드 검색, 시맨틱 검색, 퀴즈 생성 등 다양한 학습 기능을 제공합니다.

### 주요 기능

- **AI 기반 Q&A**: PDF 내용에 대한 자연어 질문-답변
- **스마트 검색**: FTS5 전문 검색 + Qdrant 시맨틱 검색
- **요약 생성**: 장/절 단위 자동 요약
- **퀴즈 생성**: 학습 내용 기반 자동 문제 출제
- **대화 컨텍스트**: 이전 대화를 기억하는 연속 대화
- **Visual Elements**: 이미지/표/수식 자동 추출 및 설명

## 🏗️ 기술 스택

- **백엔드**: Python 3.8+ FastAPI
- **LLM**: vLLM + Ollama (Dual Engine with Fallback)
- **모델**: gpt-oss:20b
- **DB**: SQLite + FTS5 전문 검색
- **벡터 DB**: Qdrant (시맨틱 검색)
- **임베딩**: sentence-transformers (jhgan/ko-sroberta-multitask)
- **PDF 처리**: PyPDF2, pdfplumber, pdf2image
- **서버**: Uvicorn (ASGI)

## 📦 포함된 교재 데이터

- **건축시공학(개정판)** (ISBN: 9788000000001)
  - 568페이지, 42개 visual elements
  - TOC 구조화, FTS5 인덱싱 완료

- **목조건축 개론** (ISBN: 9788972955610)
  - 95페이지
  - TOC 구조화, FTS5 인덱싱 완료

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 설정하세요:

```bash
# 서버
DEBUG=true
HOST=0.0.0.0
PORT=7861

# 데이터베이스
DATABASE_URL=uploads/app.db

# LLM 엔진
LLM_ENGINE=vllm
ENABLE_LLM_FALLBACK=true
FALLBACK_ENGINE=ollama

# vLLM (Docker)
VLLM_BASE_URL=http://vllm_gpt:8000
VLLM_MODEL_NAME=openai/gpt-oss-20b

# Ollama (로컬)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# 임베딩
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
EMBEDDING_DEVICE=cpu
```

### 3. Ollama 모델 설치

```bash
# Ollama 설치 (https://ollama.ai)
ollama pull gpt-oss:20b
```

### 4. Qdrant 서비스 시작

```bash
# Docker로 Qdrant 실행
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/uploads/qdrant_storage:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant
```

### 5. 서버 실행

```bash
# 개발 모드
./start_server.sh

# 또는 직접 실행
python run.py

# 백그라운드 실행
./start_server_bg.sh
```

서버가 시작되면 다음 주소에서 접속 가능:
- API: http://localhost:7861
- Swagger UI: http://localhost:7861/docs
- Health Check: http://localhost:7861/health

## 📖 API 사용 예시

### PDF Q&A

```bash
curl -X POST "http://localhost:7861/api/chat/v1/response" \
  -H "Content-Type: application/json" \
  -d '{
    "userid": "user123",
    "sessionid": "session456",
    "isbn": "9788000000001",
    "query": "1장 내용 요약해줘"
  }'
```

### 스트리밍 응답

```bash
curl -X POST "http://localhost:7861/api/chat/v1/response/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "userid": "user123",
    "sessionid": "session456",
    "isbn": "9788000000001",
    "query": "건축시공의 기본 원리는?"
  }'
```

## 🛠️ 서버 관리 스크립트

```bash
./start_server.sh      # 서버 시작 (포그라운드)
./start_server_bg.sh   # 서버 시작 (백그라운드)
./restart_server.sh    # 서버 재시작
./stop_server.sh       # 서버 중지
./server_status.sh     # 서버 상태 확인
./view_logs.sh         # 로그 확인
```

## 📊 데이터베이스 구조

- **pdfs**: PDF 메타데이터
- **pdf_pages**: 페이지별 텍스트 내용
- **pdf_pages_fts**: FTS5 전문 검색 인덱스
- **toc_entries**: 목차 구조 및 요약
- **visual_elements**: 이미지/표/수식 메타데이터

## 🔧 주요 설정

### LLM 엔진

시스템은 vLLM과 Ollama 두 엔진을 지원하며 자동 Fallback 기능이 있습니다:
1. vLLM 우선 시도 (빠름)
2. 실패 시 Ollama로 자동 전환 (안정적)

### 캐싱 시스템

- **전체 요청 캐싱**: 반복 질문 99.9% 속도 향상
- **LLM 캐시**: 3000개, 72시간 TTL
- **대화 요약**: 긴 대화 자동 압축
- **문맥 의존 감지**: 30+ 키워드로 점진적 상세화 지원

### 검색 전략

시스템은 질문 유형을 자동 분석하여 최적 검색 방법을 선택합니다:
- **toc**: 목차 기반 네비게이션
- **page**: 페이지 번호 직접 조회
- **summary**: 장/절 요약
- **keyword**: FTS5 키워드 검색
- **semantic**: Qdrant 벡터 검색
- **quiz**: 퀴즈 생성
- **followup**: 대화 컨텍스트 기반

## 📝 개발 가이드

상세한 개발 가이드는 `CLAUDE.md` 파일을 참조하세요.

## ⚠️ 주의사항

### 목조건축 개론 PDF

현재 DB에는 목조건축 개론 데이터가 포함되어 있지만, PDF 파일은 별도로 업로드해야 합니다:

```bash
# PDF 파일을 다음 경로에 배치
uploads/pdfs/7c172725-ec67-41d9-b468-37e2c6180086.pdf
```

### vLLM 사용

vLLM을 사용하려면 별도로 Docker 컨테이너를 실행해야 합니다:

```bash
docker run -d --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --name vllm_gpt \
  vllm/vllm-openai:latest \
  --model openai/gpt-oss-20b \
  --max-model-len 8192
```

## 📄 라이선스

이 프로젝트는 학습 목적으로 제공됩니다.

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!

---

**Developed with ❤️ for better learning**
