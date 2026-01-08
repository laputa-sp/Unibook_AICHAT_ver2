"""
Chunk Service - 텍스트 청크 분할
PDF 페이지를 검색 가능한 청크로 분할
"""
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


class ChunkService:
    """
    텍스트를 작은 청크로 분할하는 서비스

    목적:
    - LLM 컨텍스트 윈도우에 맞게 텍스트 분할
    - 임베딩 모델의 최대 입력 길이 준수
    - 의미 단위 유지 (문장, 단락 경계 고려)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        초기화

        Args:
            chunk_size: 청크당 최대 문자 수 (기본: 500)
            chunk_overlap: 청크 간 오버랩 문자 수 (기본: 50)
            min_chunk_size: 최소 청크 크기 (이보다 작으면 버림)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)

    def split_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        텍스트를 청크로 분할

        전략:
        1. 단락 우선 분할 (\\n\\n 기준)
        2. 문장 단위 분할 (. ! ? 기준)
        3. 청크 크기 조절 (overlap 적용)

        Args:
            text: 분할할 텍스트
            metadata: 청크에 포함할 메타데이터 (예: page_number, pdf_id)

        Returns:
            청크 딕셔너리 리스트
            [
                {
                    "text": "청크 텍스트",
                    "metadata": {...}
                },
                ...
            ]
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        # 1. 단락 분할
        paragraphs = self._split_by_paragraphs(text)

        # 2. 청크 생성
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 단락이 너무 길면 문장 단위로 분할
            if len(para) > self.chunk_size:
                # 현재 청크 저장
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # 긴 단락을 문장 단위로 분할
                sentences = self._split_by_sentences(para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) > self.chunk_size:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        # Overlap 처리
                        current_chunk = self._get_overlap(current_chunk) + sent
                    else:
                        current_chunk += (" " if current_chunk else "") + sent

            else:
                # 단락이 적당한 길이면 추가
                if len(current_chunk) + len(para) > self.chunk_size:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # Overlap 처리
                    current_chunk = self._get_overlap(current_chunk) + para
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para

        # 마지막 청크 추가
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 3. 메타데이터 추가 및 포맷팅
        result = []
        for i, chunk_text in enumerate(chunks):
            # 너무 작은 청크는 제외
            if len(chunk_text) < self.min_chunk_size:
                continue

            chunk_data = {
                "text": chunk_text,
                "chunk_index": i,
                "char_count": len(chunk_text),
                "metadata": metadata.copy()
            }
            result.append(chunk_data)

        return result

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        단락으로 분할 (\\n\\n 기준)

        Args:
            text: 입력 텍스트

        Returns:
            단락 리스트
        """
        # 2개 이상의 개행을 단락 구분자로 간주
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_sentences(self, text: str) -> List[str]:
        """
        문장으로 분할 (. ! ? 기준)

        한국어/영어 문장 경계 인식

        Args:
            text: 입력 텍스트

        Returns:
            문장 리스트
        """
        # 한국어/영어 문장 종결 기호
        # .!? 뒤에 공백이나 개행이 오는 경우
        sentence_endings = re.compile(r'([.!?]+[\s\n]+)')

        # 문장 분할
        parts = sentence_endings.split(text)

        sentences = []
        current = ""

        for part in parts:
            current += part
            if sentence_endings.match(part):
                sentences.append(current.strip())
                current = ""

        # 남은 텍스트
        if current.strip():
            sentences.append(current.strip())

        return [s for s in sentences if s]

    def _get_overlap(self, text: str) -> str:
        """
        텍스트 끝부분에서 오버랩 부분 추출

        Args:
            text: 입력 텍스트

        Returns:
            오버랩할 텍스트 (뒤에서 chunk_overlap 길이만큼)
        """
        if len(text) <= self.chunk_overlap:
            return text

        # 뒤에서 overlap 길이만큼 가져오되, 단어 경계 고려
        overlap_text = text[-self.chunk_overlap:]

        # 단어 중간이 아닌 공백에서 시작하도록 조정
        space_index = overlap_text.find(' ')
        if space_index > 0 and space_index < len(overlap_text) // 2:
            overlap_text = overlap_text[space_index + 1:]

        return overlap_text

    def split_page_content(
        self,
        pdf_id: str,
        page_number: int,
        content: str
    ) -> List[Dict]:
        """
        PDF 페이지 내용을 청크로 분할 (편의 메서드)

        Args:
            pdf_id: PDF ID
            page_number: 페이지 번호
            content: 페이지 내용

        Returns:
            청크 리스트 (metadata 포함)
        """
        metadata = {
            "pdf_id": pdf_id,
            "page_number": page_number
        }

        return self.split_text(content, metadata)

    def split_pages_batch(
        self,
        pdf_id: str,
        pages: List[Dict]
    ) -> List[Dict]:
        """
        여러 페이지를 한번에 청크로 분할

        Args:
            pdf_id: PDF ID
            pages: 페이지 리스트 [{"page_number": int, "content": str}, ...]

        Returns:
            모든 청크 리스트
        """
        all_chunks = []

        for page in pages:
            page_number = page.get('page_number')
            content = page.get('content', '')

            if not content or not content.strip():
                continue

            chunks = self.split_page_content(pdf_id, page_number, content)
            all_chunks.extend(chunks)

        self.logger.info(
            f"📄 PDF {pdf_id}: {len(pages)}페이지 → {len(all_chunks)}개 청크 생성"
        )

        return all_chunks


# Singleton 인스턴스
_chunk_service = None

def get_chunk_service(
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> ChunkService:
    """
    청크 서비스 싱글톤 반환

    Args:
        chunk_size: 청크당 최대 문자 수
        chunk_overlap: 청크 간 오버랩

    Returns:
        ChunkService 인스턴스
    """
    global _chunk_service
    if _chunk_service is None:
        _chunk_service = ChunkService(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return _chunk_service
