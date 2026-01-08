"""
Qdrant Service - 벡터 데이터베이스 관리
Qdrant를 이용한 시맨틱 검색 기능 제공
"""
import logging
import os
from typing import List, Dict, Optional, Union
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest
)
import numpy as np

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Qdrant 벡터 데이터베이스 서비스

    기능:
    - 컬렉션 생성/삭제
    - 벡터 저장 (upsert)
    - 시맨틱 검색
    - 필터링 검색
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        embedding_dim: int = 1024
    ):
        """
        초기화

        Args:
            host: Qdrant 서버 호스트
            port: Qdrant 서버 포트
            embedding_dim: 임베딩 벡터 차원 (BGE-M3: 1024)
        """
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)

        # Qdrant 클라이언트 초기화
        try:
            # 서버 연결 시도
            test_client = QdrantClient(host=host, port=port, timeout=2)
            # 연결 테스트
            test_client.get_collections()
            self.client = test_client
            self.logger.info(f"✅ Qdrant 서버 연결 성공: {host}:{port}")
        except Exception as e:
            self.logger.warning(f"⚠️  Qdrant 서버 연결 실패: {e}")
            # 로컬 디스크 모드로 폴백
            try:
                self.client = QdrantClient(path="./uploads/qdrant_storage")
                self.logger.info("✅ 로컬 디스크 모드로 Qdrant 실행 (./uploads/qdrant_storage)")
            except Exception as e2:
                self.logger.error(f"❌ 로컬 Qdrant 초기화 실패: {e2}")
                raise

    def get_collection_name(self, pdf_id: str) -> str:
        """
        PDF ID로부터 컬렉션 이름 생성

        Args:
            pdf_id: PDF UUID

        Returns:
            컬렉션 이름 (예: "pdf_abc123")
        """
        # UUID에서 하이픈 제거하고 소문자로
        clean_id = pdf_id.replace("-", "").lower()
        return f"pdf_{clean_id}"

    def create_collection(self, pdf_id: str, recreate: bool = False) -> bool:
        """
        PDF용 벡터 컬렉션 생성

        Args:
            pdf_id: PDF UUID
            recreate: 기존 컬렉션 삭제 후 재생성 여부

        Returns:
            생성 성공 여부
        """
        collection_name = self.get_collection_name(pdf_id)

        try:
            # 컬렉션 존재 확인
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if exists:
                if recreate:
                    self.logger.info(f"🗑️  기존 컬렉션 삭제: {collection_name}")
                    self.client.delete_collection(collection_name)
                else:
                    self.logger.info(f"✅ 컬렉션 이미 존재: {collection_name}")
                    return True

            # 컬렉션 생성
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE  # 코사인 유사도
                )
            )

            self.logger.info(f"✅ 컬렉션 생성 완료: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"❌ 컬렉션 생성 실패: {e}")
            return False

    def delete_collection(self, pdf_id: str) -> bool:
        """
        PDF 컬렉션 삭제

        Args:
            pdf_id: PDF UUID

        Returns:
            삭제 성공 여부
        """
        collection_name = self.get_collection_name(pdf_id)

        try:
            self.client.delete_collection(collection_name)
            self.logger.info(f"🗑️  컬렉션 삭제 완료: {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ 컬렉션 삭제 실패: {e}")
            return False

    def upsert_chunks(
        self,
        pdf_id: str,
        chunks: List[Dict],
        embeddings: List[np.ndarray]
    ) -> bool:
        """
        청크와 임베딩을 Qdrant에 저장

        Args:
            pdf_id: PDF UUID
            chunks: 청크 리스트 [{"text": str, "metadata": {...}}, ...]
            embeddings: 임베딩 벡터 리스트

        Returns:
            저장 성공 여부
        """
        if len(chunks) != len(embeddings):
            self.logger.error("청크 개수와 임베딩 개수가 다릅니다")
            return False

        collection_name = self.get_collection_name(pdf_id)

        # 컬렉션 생성 (없으면)
        self.create_collection(pdf_id, recreate=False)

        try:
            # PointStruct 리스트 생성
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # UUID 생성
                point_id = str(uuid.uuid4())

                # 페이로드 (메타데이터 + 텍스트)
                payload = {
                    "text": chunk["text"],
                    "pdf_id": pdf_id,
                    "chunk_index": chunk.get("chunk_index", i),
                    "char_count": chunk.get("char_count", len(chunk["text"])),
                }

                # 청크 메타데이터 병합
                if "metadata" in chunk:
                    payload.update(chunk["metadata"])

                # 벡터를 리스트로 변환
                vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )

            # 배치 업서트
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            self.logger.info(f"✅ {len(points)}개 청크 저장 완료: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"❌ 청크 저장 실패: {e}")
            return False

    def search(
        self,
        pdf_id: str,
        query_vector: np.ndarray,
        limit: int = 5,
        score_threshold: float = 0.5,
        page_filter: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        시맨틱 검색

        Args:
            pdf_id: PDF UUID
            query_vector: 쿼리 임베딩 벡터
            limit: 반환할 최대 결과 수
            score_threshold: 최소 유사도 점수 (0~1)
            page_filter: 검색할 페이지 번호 리스트 (None이면 전체)

        Returns:
            검색 결과 리스트
            [
                {
                    "text": "...",
                    "page_number": 10,
                    "score": 0.85,
                    ...
                },
                ...
            ]
        """
        collection_name = self.get_collection_name(pdf_id)

        try:
            # 필터 구성
            search_filter = None
            if page_filter:
                # 페이지 번호 필터링
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="page_number",
                            match=MatchValue(value=page)
                        )
                        for page in page_filter
                    ]
                )

            # 벡터를 리스트로 변환
            vector = query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector

            # 검색 실행
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold
            )

            # 결과 포맷팅
            results = []
            for hit in search_result:
                result = {
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "page_number": hit.payload.get("page_number"),
                    "chunk_index": hit.payload.get("chunk_index"),
                }
                results.append(result)

            self.logger.info(f"🔍 검색 완료: {len(results)}개 결과 (임계값: {score_threshold})")
            return results

        except Exception as e:
            self.logger.error(f"❌ 검색 실패: {e}")
            return []

    def get_collection_stats(self, pdf_id: str) -> Dict:
        """
        컬렉션 통계 조회

        Args:
            pdf_id: PDF UUID

        Returns:
            통계 정보 {"vectors_count": int, "segments_count": int, ...}
        """
        collection_name = self.get_collection_name(pdf_id)

        try:
            info = self.client.get_collection(collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
        except Exception as e:
            self.logger.error(f"❌ 통계 조회 실패: {e}")
            return {}

    def collection_exists(self, pdf_id: str) -> bool:
        """
        컬렉션 존재 여부 확인

        Args:
            pdf_id: PDF UUID

        Returns:
            존재 여부
        """
        collection_name = self.get_collection_name(pdf_id)

        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            self.logger.error(f"❌ 컬렉션 확인 실패: {e}")
            return False


# Singleton 인스턴스
_qdrant_service = None

def get_qdrant_service(
    host: str = None,
    port: int = None
) -> QdrantService:
    """
    Qdrant 서비스 싱글톤 반환

    Args:
        host: Qdrant 호스트 (None이면 환경변수 사용)
        port: Qdrant 포트 (None이면 환경변수 사용)

    Returns:
        QdrantService 인스턴스
    """
    global _qdrant_service
    if _qdrant_service is None:
        # 환경변수에서 설정 가져오기
        host = host or os.getenv("QDRANT_HOST", "localhost")
        port = port or int(os.getenv("QDRANT_PORT", "6333"))

        _qdrant_service = QdrantService(host=host, port=port)
    return _qdrant_service
