# 빠른 시작 가이드

이 문서는 Unibook AI Chat ver2를 **처음 설치하고 실행하는 방법**을 단계별로 설명합니다.

## ⚡ 5분 안에 시작하기

### 1. 저장소 클론

```bash
git clone https://github.com/laputa-sp/Unibook_AICHAT_ver2.git
cd Unibook_AICHAT_ver2
```

### 2. 환경 변수 설정

```bash
# .env.example을 복사하여 .env 생성
cp .env.example .env
```

**⚠️ 중요: `.env` 파일을 열고 다음 설정을 수정하세요:**

```bash
# .env 파일 수정
nano .env
# 또는
vi .env
```

**필수 수정 사항:**

```bash
# vLLM 설정 (본인의 Docker 환경에 맞게)
VLLM_BASE_URL=http://vllm_gpt:8000    # Docker 네트워크 내
# 또는
VLLM_BASE_URL=http://localhost:8000   # 호스트에서 실행 시

# Qdrant 설정 (본인의 Docker 환경에 맞게)
QDRANT_HOST=localhost                  # 또는 qdrant (컨테이너명)
QDRANT_PORT=6333
```

### 3. 가상환경 및 패키지 설치

```bash
# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate              # Linux/Mac
# 또는
venv\Scripts\activate                 # Windows

# 패키지 설치
pip install -r requirements.txt
```

**예상 소요 시간**: 2-3분

### 4. Docker 서비스 준비

이 시스템은 다음 Docker 컨테이너들이 **사전에 실행 중이어야** 합니다:

#### vLLM 컨테이너 실행

```bash
docker run -d --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --name vllm_gpt \
  vllm/vllm-openai:latest \
  --model openai/gpt-oss-20b \
  --max-model-len 8192
```

#### Qdrant 컨테이너 실행

```bash
docker run -d \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/uploads/qdrant_storage:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant
```

**컨테이너 상태 확인:**

```bash
docker ps | grep -E "vllm|qdrant"
```

### 5. 서버 실행

```bash
./start_server.sh
```

**예상 출력:**
```
================================
📚 PDF AI 챗봇 서버 시작
================================

포트: 7861 (또는 .env에서 설정한 포트)
로그: 터미널에 직접 출력
중지: Ctrl+C

================================

✅ 가상환경 활성화 중...
🧹 Python 캐시 삭제 중...

🚀 서버 시작 중...

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7861 (Press CTRL+C to quit)
```

### 6. 동작 확인

**새 터미널을 열고:**

```bash
# Health check
curl http://localhost:7861/health
```

**예상 응답:**
```json
{
  "status": "healthy",
  "database": "connected",
  "ollama": "connected",
  "qdrant": "connected"
}
```

### 7. 웹 UI로 테스트

1. `vllm_chat.html` 파일을 텍스트 에디터로 엽니다.
2. API URL을 수정합니다:
   - **482줄**: `http://localhost:7861/api/pdf/list`
   - **654줄**: `http://localhost:7861/api/chat/v1/response/stream`
3. 브라우저에서 `vllm_chat.html` 파일을 엽니다.
4. 도서를 선택하고 질문을 입력하세요!

## 🎯 포함된 데이터

시스템에는 다음 교재 데이터가 포함되어 있습니다:

- **건축시공학(개정판)** (1-50페이지)
- **목조건축 개론** (전체 95페이지)

## ❓ 문제 해결

### ".env 파일이 없습니다" 에러

**해결:**
```bash
cp .env.example .env
```

그런 다음 `.env` 파일을 열어서 Docker 설정을 수정하세요.

### "가상환경이 없습니다" 에러

**해결:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### "Connection refused" 에러

**원인**: vLLM 또는 Qdrant 컨테이너가 실행되지 않음

**해결:**
```bash
# 컨테이너 상태 확인
docker ps | grep -E "vllm|qdrant"

# 컨테이너 시작
docker start vllm_gpt qdrant

# 로그 확인
docker logs vllm_gpt
docker logs qdrant
```

### "ModuleNotFoundError" 에러

**원인**: 패키지가 설치되지 않음

**해결:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 포트 충돌 에러

**원인**: 7861 포트가 이미 사용 중

**해결:**
```bash
# .env 파일에서 포트 변경
PORT=8000  # 또는 다른 포트

# 서버 재시작
./restart_server.sh
```

## 📚 추가 문서

- **README.md**: 전체 프로젝트 개요
- **DOCKER_SETUP.md**: Docker 네트워크 상세 설정
- **TEST_GUIDE.md**: 테스트 시나리오 및 방법
- **CLAUDE.md**: 개발자 가이드

## 🆘 도움이 필요하신가요?

이슈를 등록해주세요:
https://github.com/laputa-sp/Unibook_AICHAT_ver2/issues

---

**Happy Coding! 🚀**
