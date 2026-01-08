"""
대화 로거 - 세션별 대화 내용을 JSON 파일로 저장
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConversationLogger:
    """세션별 대화 내용을 파일로 저장"""

    def __init__(self, log_dir: str = "uploads/conversations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📝 대화 로거 초기화: {self.log_dir}")

    def _get_log_file(self, session_id: str) -> Path:
        """세션 ID로 로그 파일 경로 생성"""
        # 날짜별 폴더 생성 (YYYY-MM-DD)
        date_folder = datetime.now().strftime("%Y-%m-%d")
        folder = self.log_dir / date_folder
        folder.mkdir(parents=True, exist_ok=True)

        # 세션 ID로 파일명 생성
        return folder / f"{session_id}.json"

    async def log_conversation(
        self,
        session_id: str,
        user_id: str,
        isbn: str,
        query: str,
        response: str,
        metadata: Dict[str, Any] = None
    ):
        """대화 내용을 파일에 저장"""
        try:
            log_file = self._get_log_file(session_id)

            # 기존 로그 읽기 (있으면)
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "isbn": isbn,
                    "created_at": datetime.now().isoformat(),
                    "conversations": []
                }

            # 새 대화 추가
            conversation = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response": response,
                "metadata": metadata or {}
            }
            data["conversations"].append(conversation)
            data["updated_at"] = datetime.now().isoformat()

            # 파일에 저장
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 대화 저장: {session_id} ({len(data['conversations'])}개)")

        except Exception as e:
            logger.error(f"❌ 대화 저장 실패: {e}")

    async def get_conversation(self, session_id: str) -> Dict[str, Any]:
        """세션의 대화 내용 조회"""
        try:
            log_file = self._get_log_file(session_id)

            if not log_file.exists():
                return None

            with open(log_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"❌ 대화 조회 실패: {e}")
            return None

    async def create_session(
        self,
        session_id: str,
        user_id: str,
        isbn: str,
        system_message: str = None
    ):
        """새 세션 생성 (시스템 메시지 포함)"""
        try:
            log_file = self._get_log_file(session_id)

            # 새 세션 데이터
            data = {
                "session_id": session_id,
                "user_id": user_id,
                "isbn": isbn,
                "created_at": datetime.now().isoformat(),
                "conversations": []
            }

            # 시스템 메시지 추가 (있으면)
            if system_message:
                data["conversations"].append({
                    "timestamp": datetime.now().isoformat(),
                    "query": None,
                    "response": system_message,
                    "metadata": {"type": "system"}
                })

            data["updated_at"] = datetime.now().isoformat()

            # 파일에 저장
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"✨ 새 세션 생성: {session_id}")
            return data

        except Exception as e:
            logger.error(f"❌ 세션 생성 실패: {e}")
            return None


# 싱글톤 인스턴스
_conversation_logger = None


def get_conversation_logger() -> ConversationLogger:
    """대화 로거 싱글톤 반환"""
    global _conversation_logger
    if _conversation_logger is None:
        _conversation_logger = ConversationLogger()
    return _conversation_logger
