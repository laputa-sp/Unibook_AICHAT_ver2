# 설치 체크리스트

새로 설치하시는 분들을 위한 체크리스트입니다. 각 단계를 완료하면 체크(✅)하세요.

## 📋 설치 전 준비사항

- [ ] Python 3.8 이상 설치됨
- [ ] Docker 설치됨
- [ ] NVIDIA GPU 및 CUDA 설치됨 (vLLM 사용 시)
- [ ] Git 설치됨

## 🔧 설치 단계

### 1단계: 저장소 클론
```bash
git clone https://github.com/laputa-sp/Unibook_AICHAT_ver2.git
cd Unibook_AICHAT_ver2
```
- [ ] 저장소 클론 완료

### 2단계: 환경 설정
```bash
cp .env.example .env
nano .env  # 또는 원하는 에디터
```

**필수 수정 항목:**
- [ ] `VLLM_BASE_URL` 수정 (Docker 환경에 맞게)
- [ ] `QDRANT_HOST` 수정 (Docker 환경에 맞게)
- [ ] `QDRANT_PORT` 확인
- [ ] `PORT` 확인 (기본값: 7861)

### 3단계: Python 환경
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```
- [ ] 가상환경 생성 완료
- [ ] 가상환경 활성화 완료
- [ ] 패키지 설치 완료 (2-3분 소요)

### 4단계: Docker 서비스

#### vLLM 컨테이너
```bash
docker run -d --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --name vllm_gpt \
  vllm/vllm-openai:latest \
  --model openai/gpt-oss-20b \
  --max-model-len 8192
```
- [ ] vLLM 컨테이너 실행 완료
- [ ] `docker ps | grep vllm` 확인

#### Qdrant 컨테이너
```bash
docker run -d \
  -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/uploads/qdrant_storage:/qdrant/storage \
  --name qdrant \
  qdrant/qdrant
```
- [ ] Qdrant 컨테이너 실행 완료
- [ ] `docker ps | grep qdrant` 확인

### 5단계: 서버 실행
```bash
./start_server.sh
```
- [ ] 서버 시작 성공
- [ ] 로그에 "Application startup complete" 표시됨

### 6단계: 동작 확인
```bash
curl http://localhost:7861/health
```
- [ ] Health check 성공 (status: healthy)
- [ ] Database 연결 확인
- [ ] Qdrant 연결 확인

### 7단계: 웹 UI 테스트
1. `vllm_chat.html` 파일 수정
   - [ ] 482줄 API URL 수정
   - [ ] 654줄 API URL 수정
2. 브라우저로 열기
   - [ ] 브라우저에서 파일 열림
   - [ ] PDF 목록 로드됨
3. 테스트
   - [ ] 도서 선택 가능
   - [ ] 질문 입력 가능
   - [ ] 답변 정상 수신

## ✅ 설치 완료!

모든 체크박스에 체크하셨다면 설치가 완료된 것입니다! 🎉

## 🐛 문제 발생 시

각 단계에서 문제가 발생하면:

1. **QUICKSTART.md**의 문제 해결 섹션 확인
2. **TEST_GUIDE.md**의 문제 해결 가이드 확인
3. **DOCKER_SETUP.md**의 상세 설정 확인
4. 그래도 해결 안 되면 이슈 등록: https://github.com/laputa-sp/Unibook_AICHAT_ver2/issues

## 📚 다음 단계

설치가 완료되었다면:
- **TEST_GUIDE.md**: 다양한 테스트 시나리오 시도
- **CLAUDE.md**: 개발 가이드 참고
- **DOCKER_SETUP.md**: Docker 네트워크 고급 설정

---

Happy Learning! 📖
