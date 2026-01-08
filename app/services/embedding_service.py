"""
Embedding Service - BGE-M3 기반 텍스트 임베딩
BGE-M3는 다국어 지원이 우수한 BAAI의 임베딩 모델
"""
import logging
from typing import List, Union
import numpy as np
from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    BGE-M3 기반 임베딩 서비스

    BGE-M3 특징:
    - 다국어 지원 (100+ languages)
    - 긴 컨텍스트 지원 (최대 8192 tokens)
    - Dense + Sparse + ColBERT 하이브리드 검색 지원
    - 한국어 성능 우수
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        """
        초기화

        Args:
            model_name: 모델명 (기본: BAAI/bge-m3)
            use_fp16: FP16 사용 여부 (GPU 메모리 절약)
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.use_fp16 = use_fp16
        self.embedding_dim = 1024  # BGE-M3의 임베딩 차원

    def load_model(self):
        """모델 로드 (lazy loading)"""
        if self.model is None:
            self.logger.info(f"📥 BGE-M3 모델 로딩 중... ({self.model_name})")
            try:
                self.model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self.use_fp16
                )
                self.logger.info("✅ BGE-M3 모델 로드 완료!")
            except Exception as e:
                self.logger.error(f"❌ 모델 로드 실패: {e}")
                raise

    def embed_text(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트
            max_length: 최대 토큰 길이 (기본: 512, 최대: 8192)

        Returns:
            numpy array (1024 dim)
        """
        self.load_model()

        if not text or not text.strip():
            self.logger.warning("빈 텍스트 입력, 제로 벡터 반환")
            return np.zeros(self.embedding_dim)

        try:
            # BGE-M3 encode
            # return_dense=True: Dense 벡터만 반환 (기본 시맨틱 검색용)
            # return_sparse=False, return_colbert_vecs=False: Sparse/ColBERT 미사용
            embeddings = self.model.encode(
                [text],
                batch_size=1,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )

            # Dense 임베딩 추출
            embedding_vector = embeddings['dense_vecs'][0]

            # numpy array로 변환 (이미 numpy일 수 있음)
            if not isinstance(embedding_vector, np.ndarray):
                embedding_vector = np.array(embedding_vector)

            return embedding_vector

        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            # 에러 시 제로 벡터 반환
            return np.zeros(self.embedding_dim)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        배치 텍스트 임베딩 (효율적)

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기 (기본: 32)
            max_length: 최대 토큰 길이
            show_progress: 진행률 표시 여부

        Returns:
            numpy array 리스트 (각 1024 dim)
        """
        self.load_model()

        if not texts:
            return []

        # 빈 텍스트 필터링
        processed_texts = []
        indices_map = []  # 원본 인덱스 매핑

        for i, text in enumerate(texts):
            if text and text.strip():
                processed_texts.append(text)
                indices_map.append(i)

        if not processed_texts:
            self.logger.warning("모든 텍스트가 비어있음, 제로 벡터 반환")
            return [np.zeros(self.embedding_dim) for _ in texts]

        try:
            self.logger.info(f"📊 배치 임베딩 생성 중... (총 {len(processed_texts)}개)")

            # BGE-M3 배치 인코딩
            # Note: show_progress_bar is not supported, use show_progress instead
            embeddings = self.model.encode(
                processed_texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )

            # Dense 벡터 추출
            dense_vecs = embeddings['dense_vecs']

            # 원본 순서대로 결과 재구성 (빈 텍스트는 제로 벡터)
            result = [np.zeros(self.embedding_dim) for _ in texts]
            for i, original_idx in enumerate(indices_map):
                result[original_idx] = dense_vecs[i]

            self.logger.info(f"✅ 배치 임베딩 완료!")
            return result

        except Exception as e:
            self.logger.error(f"배치 임베딩 실패: {e}")
            # 에러 시 제로 벡터 리스트 반환
            return [np.zeros(self.embedding_dim) for _ in texts]

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        검색 쿼리 임베딩 (별도 프롬프트 처리 가능)

        BGE-M3는 쿼리/문서 구분이 필요없으나,
        향후 모델 교체 시를 위해 별도 메서드 제공

        Args:
            query: 검색 쿼리

        Returns:
            numpy array (1024 dim)
        """
        # BGE-M3는 대칭적 모델이므로 일반 임베딩과 동일
        return self.embed_text(query, max_length=512)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        코사인 유사도 계산

        Args:
            vec1, vec2: 임베딩 벡터

        Returns:
            코사인 유사도 (-1 ~ 1)
        """
        # 제로 벡터 체크
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 코사인 유사도
        return float(np.dot(vec1, vec2) / (norm1 * norm2))


# Singleton 인스턴스 생성
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """
    임베딩 서비스 싱글톤 반환

    Returns:
        EmbeddingService 인스턴스
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
