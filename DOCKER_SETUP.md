# Docker 환경 설정 가이드

이 프로젝트는 **Docker 환경에서 실행**되도록 설계되었습니다.

## 📋 사전 요구사항

다음 Docker 컨테이너들이 실행 중이어야 합니다:
1. **vLLM** - LLM 추론 서버
2. **Qdrant** - 벡터 데이터베이스

## 🐳 Docker 네트워크 구성

### 옵션 1: 동일 Docker 네트워크 (권장)

모든 컨테이너를 동일한 Docker 네트워크에서 실행하는 경우:

```bash
# Docker 네트워크 생성
docker network create unibook-network

# vLLM 컨테이너 (네트워크 연결)
docker run -d --gpus all \
  --network unibook-network \
  --name vllm_gpt \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model openai/gpt-oss-20b \
  --max-model-len 8192

# Qdrant 컨테이너 (네트워크 연결)
docker run -d \
  --network unibook-network \
  --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Python 앱도 동일 네트워크에서 실행
docker run -d \
  --network unibook-network \
  --name unibook_api \
  -p 7861:7861 \
  -v $(pwd):/app \
  -w /app \
  python:3.9 \
  bash -c "pip install -r requirements.txt && python run.py"
```

**.env 설정**:
```bash
VLLM_BASE_URL=http://vllm_gpt:8000    # 컨테이너 이름 사용
QDRANT_HOST=qdrant                     # 컨테이너 이름 사용
QDRANT_PORT=6333
```

### 옵션 2: 호스트 네트워크

Python 앱을 호스트에서 실행하고 Docker 서비스에 접근하는 경우:

```bash
# vLLM 컨테이너
docker run -d --gpus all \
  --name vllm_gpt \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model openai/gpt-oss-20b \
  --max-model-len 8192

# Qdrant 컨테이너
docker run -d \
  --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Python 앱은 호스트에서 실행
python run.py
```

**.env 설정**:
```bash
VLLM_BASE_URL=http://localhost:8000   # localhost 사용
QDRANT_HOST=localhost                  # localhost 사용
QDRANT_PORT=6333
```

## 🔍 연결 확인

### vLLM 상태 확인
```bash
curl http://localhost:8000/v1/models
# 또는 Docker 네트워크 내에서
curl http://vllm_gpt:8000/v1/models
```

### Qdrant 상태 확인
```bash
curl http://localhost:6333/collections
# 또는 Docker 네트워크 내에서
curl http://qdrant:6333/collections
```

### 앱 Health Check
```bash
curl http://localhost:7861/health
```

## 🛠️ 포트 충돌 해결

기본 포트가 이미 사용 중인 경우:

```bash
# vLLM을 8001 포트로 실행
docker run -d --gpus all \
  --name vllm_gpt \
  -p 8001:8000 \
  ...

# Qdrant를 6334 포트로 실행
docker run -d \
  --name qdrant \
  -p 6334:6333 \
  ...
```

**.env 수정**:
```bash
VLLM_BASE_URL=http://localhost:8001
QDRANT_PORT=6334
```

## 🐛 문제 해결

### "Connection refused" 에러

**원인**: Docker 컨테이너가 실행 중이 아니거나 네트워크 설정 오류

**해결**:
```bash
# 컨테이너 상태 확인
docker ps -a | grep -E "vllm|qdrant"

# 컨테이너 재시작
docker restart vllm_gpt qdrant

# 로그 확인
docker logs vllm_gpt
docker logs qdrant
```

### "Cannot find container" 에러

**원인**: `.env`의 호스트명이 Docker 네트워크에 맞지 않음

**해결**:
- Docker 네트워크 내: 컨테이너 이름 사용 (`vllm_gpt`, `qdrant`)
- 호스트에서 실행: `localhost` 사용

### GPU 관련 에러

**원인**: NVIDIA Container Toolkit 미설치

**해결**:
```bash
# NVIDIA Container Toolkit 설치 확인
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# 설치 방법: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

## 📚 참고

- 이 프로젝트는 로컬 테스트 환경(`localhost`)을 기반으로 개발되었습니다.
- 실제 배포 시에는 본인의 Docker 환경에 맞게 설정을 조정해야 합니다.
- Docker Compose를 사용하면 더 쉽게 관리할 수 있습니다. (추후 추가 예정)
