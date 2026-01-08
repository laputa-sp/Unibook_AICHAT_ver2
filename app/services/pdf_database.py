"""
PDF Database Service
Handles database operations for PDF management
"""
import aiosqlite
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy imports for Qdrant (optional dependency)
_qdrant_service = None
_embedding_service = None

def _get_qdrant_service():
    """Lazy load Qdrant service"""
    global _qdrant_service
    if _qdrant_service is None and settings.QDRANT_ENABLED:
        from app.services.qdrant_service import get_qdrant_service
        _qdrant_service = get_qdrant_service()
    return _qdrant_service

def _get_embedding_service():
    """Lazy load Embedding service"""
    global _embedding_service
    if _embedding_service is None:
        from app.services.embedding_service import get_embedding_service
        _embedding_service = get_embedding_service()
    return _embedding_service


class PDFDatabase:
    """Database service for PDF management"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DATABASE_URL
        self.logger = logging.getLogger(__name__)

    async def init_database(self):
        """Initialize database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Create PDFs table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS pdfs (
                    id TEXT PRIMARY KEY,
                    isbn TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    has_toc INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create index on ISBN
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_pdfs_isbn ON pdfs(isbn)
            ''')

            # Create PDF pages table
            await db.execute('''
                CREATE TABLE IF NOT EXISTS pdf_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    content TEXT,
                    word_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (pdf_id) REFERENCES pdfs(id) ON DELETE CASCADE,
                    UNIQUE(pdf_id, page_number)
                )
            ''')

            # Create index on pdf_id and page_number
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_pages_pdf_id ON pdf_pages(pdf_id)
            ''')

            # Create FTS5 table for full-text search
            await db.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS pdf_pages_fts USING fts5(
                    pdf_id UNINDEXED,
                    page_number UNINDEXED,
                    content,
                    content=pdf_pages,
                    content_rowid=id
                )
            ''')

            # Create triggers to keep FTS table in sync
            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS pdf_pages_ai AFTER INSERT ON pdf_pages BEGIN
                    INSERT INTO pdf_pages_fts(rowid, pdf_id, page_number, content)
                    VALUES (new.id, new.pdf_id, new.page_number, new.content);
                END
            ''')

            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS pdf_pages_ad AFTER DELETE ON pdf_pages BEGIN
                    DELETE FROM pdf_pages_fts WHERE rowid = old.id;
                END
            ''')

            await db.execute('''
                CREATE TRIGGER IF NOT EXISTS pdf_pages_au AFTER UPDATE ON pdf_pages BEGIN
                    UPDATE pdf_pages_fts SET content = new.content
                    WHERE rowid = new.id;
                END
            ''')

            await db.commit()
            self.logger.info("Database initialized successfully")

    async def register_pdf(
        self,
        isbn: str,
        title: str,
        file_path: str,
        page_count: int,
        file_size: int
    ) -> str:
        """
        Register a new PDF in the database

        Returns:
            PDF ID (UUID)
        """
        pdf_id = str(uuid.uuid4())

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT INTO pdfs (id, isbn, title, file_path, page_count, file_size)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (pdf_id, isbn, title, file_path, page_count, file_size))

            await db.commit()

        self.logger.info(f"Registered PDF: {pdf_id} - {title}")
        return pdf_id

    async def get_pdf_by_isbn(self, isbn: str) -> Optional[Dict]:
        """Get PDF information by ISBN"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM pdfs WHERE isbn = ?
            ''', (isbn,))

            row = await cursor.fetchone()

            if row:
                return dict(row)

        return None

    async def get_pdf_by_id(self, pdf_id: str) -> Optional[Dict]:
        """Get PDF information by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM pdfs WHERE id = ?
            ''', (pdf_id,))

            row = await cursor.fetchone()

            if row:
                return dict(row)

        return None

    async def get_all_pdfs(self) -> List[Dict]:
        """Get all PDFs"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM pdfs ORDER BY created_at DESC
            ''')

            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete_pdf(self, isbn: str) -> bool:
        """Delete PDF by ISBN"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                DELETE FROM pdfs WHERE isbn = ?
            ''', (isbn,))

            await db.commit()

            return cursor.rowcount > 0

    async def save_page_content(
        self,
        pdf_id: str,
        page_number: int,
        content: str,
        word_count: int = 0
    ):
        """Save page content to database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                INSERT OR REPLACE INTO pdf_pages (pdf_id, page_number, content, word_count)
                VALUES (?, ?, ?, ?)
            ''', (pdf_id, page_number, content, word_count))

            await db.commit()

    async def save_pages_batch(self, pdf_id: str, pages: List[Dict]):
        """Save multiple pages in a batch"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executemany('''
                INSERT OR REPLACE INTO pdf_pages (pdf_id, page_number, content, word_count)
                VALUES (?, ?, ?, ?)
            ''', [(pdf_id, p['page_number'], p['content'], p['word_count']) for p in pages])

            await db.commit()

        self.logger.info(f"Saved {len(pages)} pages for PDF {pdf_id}")

    async def get_page_content(self, pdf_id: str, page_number: int) -> Optional[Dict]:
        """Get page content"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM pdf_pages
                WHERE pdf_id = ? AND page_number = ?
            ''', (pdf_id, page_number))

            row = await cursor.fetchone()

            if row:
                return dict(row)

        return None

    async def get_pages_range(
        self,
        pdf_id: str,
        start_page: int,
        end_page: int
    ) -> List[Dict]:
        """Get pages in a range"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT * FROM pdf_pages
                WHERE pdf_id = ? AND page_number >= ? AND page_number <= ?
                ORDER BY page_number
            ''', (pdf_id, start_page, end_page))

            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def search_pages(
        self,
        pdf_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict]:
        """Full-text search in PDF pages"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT p.*, rank
                FROM pdf_pages p
                JOIN pdf_pages_fts fts ON p.id = fts.rowid
                WHERE fts.content MATCH ? AND p.pdf_id = ?
                ORDER BY rank
                LIMIT ?
            ''', (query, pdf_id, limit))

            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def update_pdf_title(self, isbn: str, title: str) -> bool:
        """Update PDF title"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('''
                UPDATE pdfs SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE isbn = ?
            ''', (title, isbn))

            await db.commit()

            return cursor.rowcount > 0

    async def get_pdf_stats(self, pdf_id: str) -> Dict:
        """Get PDF statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            # Get page count with content
            cursor = await db.execute('''
                SELECT COUNT(*) as pages_with_content
                FROM pdf_pages
                WHERE pdf_id = ? AND LENGTH(content) > 0
            ''', (pdf_id,))

            row = await cursor.fetchone()
            pages_with_content = row[0] if row else 0

            # Get total word count
            cursor = await db.execute('''
                SELECT SUM(word_count) as total_words
                FROM pdf_pages
                WHERE pdf_id = ?
            ''', (pdf_id,))

            row = await cursor.fetchone()
            total_words = row[0] if row and row[0] else 0

            return {
                'pages_with_content': pages_with_content,
                'total_words': total_words
            }

    async def search_pdf_content(
        self,
        pdf_id: str,
        core_keywords: List[List[str]],
        sub_keywords: List[str],
        page_numbers: Optional[List[int]] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Advanced FTS search with core and sub keywords

        Replicates pdfService.fetchResults() from Node.js

        Args:
            pdf_id: PDF ID
            core_keywords: List of keyword groups (AND across groups, OR within group)
                Example: [["자기효능감", "self-efficacy"], ["밴듀라", "Bandura"]]
            sub_keywords: List of optional keywords (OR, boosts score)
            page_numbers: Optional list of page numbers to restrict search
            limit: Maximum results

        Returns:
            List of page content dictionaries with page_number and content
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Build FTS query
            # Core keywords: AND across groups, OR within group
            # Example: (자기효능감 OR self-efficacy) AND (밴듀라 OR Bandura)

            def has_korean(text: str) -> bool:
                """Check if text contains Korean characters"""
                return any('\uAC00' <= char <= '\uD7A3' for char in text)

            def format_keyword(kw: str) -> str:
                """Format keyword for FTS5 MATCH query
                - Korean: use prefix wildcard (내향형 → 내향*)
                - English: use exact phrase ("self-efficacy")
                """
                if has_korean(kw):
                    # Use prefix matching for Korean (FTS5 tokenization issue)
                    # Remove common suffixes and add wildcard
                    return f"{kw}*"
                else:
                    # Use exact phrase for English
                    return f'"{kw}"'

            fts_query_parts = []

            # Add core keywords (must all match)
            for kw_group in core_keywords:
                if kw_group:
                    # OR within group
                    group_query = ' OR '.join([format_keyword(kw) for kw in kw_group])
                    fts_query_parts.append(f"({group_query})")

            # Build final core query
            if fts_query_parts:
                core_query = ' AND '.join(fts_query_parts)
            else:
                # No core keywords - match all
                core_query = '*'

            # Debug: Log the FTS query
            self.logger.info(f"🔍 FTS5 MATCH query: {core_query}")

            # Add sub keywords (boost if present, but not required)
            # In FTS5, we can't easily do "boost but not require"
            # So we'll do a second query if no results with core only

            # Build WHERE clause for page filtering
            where_clauses = ['p.pdf_id = ?']
            params = [pdf_id]

            if page_numbers:
                placeholders = ','.join(['?' for _ in page_numbers])
                where_clauses.append(f'p.page_number IN ({placeholders})')
                params.extend(page_numbers)

            where_clause = ' AND '.join(where_clauses)

            # Execute main query with core keywords
            sql = f'''
                SELECT DISTINCT p.page_number, p.content
                FROM pdf_pages p
                JOIN pdf_pages_fts fts ON p.id = fts.rowid
                WHERE {where_clause} AND fts.content MATCH ?
                ORDER BY p.page_number
                LIMIT ?
            '''

            params.append(core_query)
            params.append(limit)

            # Debug: Log SQL and params
            sql_oneline = ' '.join(sql.split())
            self.logger.info(f"📝 SQL: {sql_oneline[:200]}")
            self.logger.info(f"📝 Params ({len(params)}): first='{params[0]}', last 2={params[-2:]}")
            self.logger.info(f"📝 Page range: {page_numbers[:3] if page_numbers else 'ALL'}...{page_numbers[-3:] if page_numbers and len(page_numbers) > 3 else ''}")

            try:
                cursor = await db.execute(sql, params)
                rows = await cursor.fetchall()
                self.logger.info(f"📊 Query returned {len(rows)} rows")

                if rows:
                    self.logger.info(f"✅ Returning {len(rows)} results: pages {[r['page_number'] for r in rows]}")
                    return [dict(row) for row in rows]

                self.logger.warning(f"⚠️ No results found! Checking if fallback needed...")

                # If no results and we have sub keywords, try a broader search
                if sub_keywords:
                    # Try with sub keywords only
                    sub_query = ' OR '.join([format_keyword(kw) for kw in sub_keywords])
                    params_sub = params[:-2] + [sub_query, limit]

                    cursor = await db.execute(sql, params_sub)
                    rows = await cursor.fetchall()

                    return [dict(row) for row in rows]

                return []

            except Exception as e:
                self.logger.error(f"❌ FTS search error: {e}", exc_info=True)
                self.logger.info("🔄 Falling back to LIKE search...")
                # Fallback to LIKE search
                return await self._fallback_like_search(
                    db, pdf_id, core_keywords, page_numbers, limit
                )

    async def _fallback_like_search(
        self,
        db,
        pdf_id: str,
        core_keywords: List[List[str]],
        page_numbers: Optional[List[int]],
        limit: int
    ) -> List[Dict]:
        """Fallback to LIKE search if FTS fails"""
        db.row_factory = aiosqlite.Row

        # Build LIKE conditions for core keywords
        like_conditions = []
        params = [pdf_id]

        for kw_group in core_keywords:
            if kw_group:
                # At least one keyword in group must match
                group_conditions = []
                for kw in kw_group:
                    group_conditions.append('p.content LIKE ?')
                    params.append(f'%{kw}%')

                like_conditions.append(f"({' OR '.join(group_conditions)})")

        where_clause = 'p.pdf_id = ?'
        if like_conditions:
            where_clause += ' AND ' + ' AND '.join(like_conditions)

        if page_numbers:
            placeholders = ','.join(['?' for _ in page_numbers])
            where_clause += f' AND p.page_number IN ({placeholders})'
            params.extend(page_numbers)

        params.append(limit)

        sql = f'''
            SELECT DISTINCT p.page_number, p.content
            FROM pdf_pages p
            WHERE {where_clause}
            ORDER BY p.page_number
            LIMIT ?
        '''

        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()

        return [dict(row) for row in rows]

    async def semantic_search_pdf_content(
        self,
        pdf_id: str,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5,
        page_numbers: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        시맨틱 검색 (Qdrant 기반)

        기존 FTS5 키워드 검색을 대체하는 의미 기반 검색
        - 동의어 자동 검색
        - 문맥 이해
        - 다국어 지원

        Args:
            pdf_id: PDF ID
            query: 사용자 질문 (자연어)
            limit: 최대 결과 수
            score_threshold: 최소 유사도 점수 (0~1)
            page_numbers: 검색할 페이지 번호 리스트

        Returns:
            검색 결과 리스트 [{"page_number": int, "content": str, "score": float}, ...]
        """
        if not settings.QDRANT_ENABLED:
            self.logger.warning("Qdrant가 비활성화됨, FTS5 폴백 사용")
            # FTS5로 폴백 (키워드만 사용)
            keywords = query.split()[:3]  # 간단히 단어 분할
            core_keywords = [[kw] for kw in keywords]
            return await self.search_pdf_content(
                pdf_id=pdf_id,
                core_keywords=core_keywords,
                sub_keywords=[],
                page_numbers=page_numbers,
                limit=limit
            )

        try:
            # Qdrant 및 Embedding 서비스 가져오기
            qdrant = _get_qdrant_service()
            embedder = _get_embedding_service()

            if not qdrant or not embedder:
                raise Exception("Qdrant 또는 Embedding 서비스 초기화 실패")

            # 컬렉션 존재 확인
            if not qdrant.collection_exists(pdf_id):
                self.logger.warning(f"PDF {pdf_id}의 벡터 컬렉션이 없음, 빈 결과 반환")
                return []

            # 쿼리 임베딩 생성
            self.logger.info(f"🔍 시맨틱 검색 시작: '{query[:50]}...'")
            query_embedding = embedder.get_query_embedding(query)

            # Qdrant 검색
            results = qdrant.search(
                pdf_id=pdf_id,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                page_filter=page_numbers
            )

            # SQLite에서 페이지 전체 내용 가져오기 (Qdrant는 청크만 저장)
            # 페이지 번호 추출
            page_nums = list(set([r['page_number'] for r in results if r.get('page_number')]))

            if not page_nums:
                return []

            # 페이지 내용 조회
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                placeholders = ','.join(['?' for _ in page_nums])
                sql = f'''
                    SELECT page_number, content
                    FROM pdf_pages
                    WHERE pdf_id = ? AND page_number IN ({placeholders})
                    ORDER BY page_number
                '''
                cursor = await db.execute(sql, [pdf_id] + page_nums)
                rows = await cursor.fetchall()

                # 스코어 매핑 (페이지별 최고 점수)
                page_scores = {}
                for r in results:
                    pn = r.get('page_number')
                    score = r.get('score', 0)
                    if pn and (pn not in page_scores or score > page_scores[pn]):
                        page_scores[pn] = score

                # 결과 포맷팅
                final_results = []
                for row in rows:
                    final_results.append({
                        'page_number': row['page_number'],
                        'content': row['content'],
                        'score': page_scores.get(row['page_number'], 0.0)
                    })

                self.logger.info(f"✅ 시맨틱 검색 완료: {len(final_results)}개 페이지")
                return final_results

        except Exception as e:
            self.logger.error(f"❌ 시맨틱 검색 실패: {e}")
            # FTS5로 폴백
            self.logger.info("FTS5 키워드 검색으로 폴백...")
            keywords = query.split()[:3]
            core_keywords = [[kw] for kw in keywords]
            return await self.search_pdf_content(
                pdf_id=pdf_id,
                core_keywords=core_keywords,
                sub_keywords=[],
                page_numbers=page_numbers,
                limit=limit
            )

    # ========== TOC (Table of Contents) Methods ==========

    async def get_toc_entries(self, pdf_id: str) -> List[Dict]:
        """
        Get all TOC entries for a PDF

        Returns:
            List of TOC entries with title, start_page, end_page, core_summary, summary
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute('''
                SELECT id, title, start_page, end_page, core_summary, summary
                FROM toc_entries
                WHERE pdf_id = ?
                ORDER BY start_page
            ''', (pdf_id,))

            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_toc_simple(self, pdf_id: str) -> str:
        """
        Get simple TOC (title | page range) - Replicates getTocSimple()

        Returns:
            String with format: "title | page start-end" per line
        """
        entries = await self.get_toc_entries(pdf_id)

        if not entries:
            return ""

        lines = []
        for entry in entries:
            title = entry['title']
            start_page = entry['start_page']
            end_page = entry['end_page']
            lines.append(f"{title} | page {start_page}-{end_page}")

        return "\n".join(lines)

    async def get_toc_chapters_only(self, pdf_id: str) -> str:
        """
        Get TOC with chapter-level entries only (no subsections)

        Filters out entries that appear to be subsections (contains dots, 절, or indentation markers)

        Returns:
            String with format: "title | page start-end" per line (chapters only)
        """
        entries = await self.get_toc_entries(pdf_id)

        if not entries:
            return ""

        lines = []
        for entry in entries:
            title = entry['title']

            # Filter logic: skip if title looks like a subsection
            # Skip if contains: ".", "절", starts with whitespace, or has numbering like "1-1", "1.1"
            if any([
                '.' in title and any(char.isdigit() for char in title.split('.')[0]),  # "1.1", "2.3"
                '절' in title,  # "1절", "제1절"
                title.startswith((' ', '\t')),  # Indented entries
                '-' in title and len(title.split('-')) > 1 and title.split('-')[0].strip().replace('장', '').replace('제', '').isdigit(),  # "1-1", "2-3"
            ]):
                continue

            # Only include chapter-level entries
            start_page = entry['start_page']
            end_page = entry['end_page']
            lines.append(f"{title} | page {start_page}-{end_page}")

        return "\n".join(lines)

    async def get_toc_core_summary_text(self, pdf_id: str) -> str:
        """
        Get TOC with core summary keywords - Replicates getTocCoreSummaryText()

        Returns:
            String with format: "title | page start-end | keywords: keyword1, keyword2, ..."
        """
        entries = await self.get_toc_entries(pdf_id)

        if not entries:
            return ""

        lines = []
        for entry in entries:
            title = entry['title']
            start_page = entry['start_page']
            end_page = entry['end_page']
            core_summary = entry.get('core_summary', '') or ''

            # Extract first 3 keywords
            keywords = [k.strip() for k in core_summary.split(',') if k.strip()][:3]
            keywords_str = ', '.join(keywords)

            lines.append(f"{title} | page {start_page}-{end_page} | keywords : {keywords_str}")

        return "\n".join(lines)

    async def get_toc_summaries_by_page_spec(self, pdf_id: str, page_spec: str) -> str:
        """
        Get TOC summaries for specified page range - Replicates getTocSummariesByPageSpec()

        Args:
            pdf_id: PDF ID
            page_spec: Page specification like "1,2,10-20,55"

        Returns:
            Concatenated summaries from TOC entries in the page range
        """
        # Parse page spec
        pages = self._parse_page_spec(page_spec)
        if not pages:
            return ""

        # N+1 쿼리 방지: 단일 쿼리로 범위 조회
        min_page = min(pages)
        max_page = max(pages)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # 범위 내 모든 TOC 엔트리를 한 번에 조회
            cursor = await db.execute('''
                SELECT DISTINCT title, summary, start_page, end_page
                FROM toc_entries
                WHERE pdf_id = ?
                  AND NOT (end_page < ? OR start_page > ?)
                ORDER BY start_page
            ''', (pdf_id, min_page, max_page))

            rows = await cursor.fetchall()
            summaries = []
            for row in rows:
                summary_text = row['summary'] or ''
                if summary_text and summary_text != '없음':
                    summaries.append(f"[{row['title']} (페이지 {row['start_page']}-{row['end_page']})]\n{summary_text}")

            return "\n\n".join(summaries) if summaries else ""

    def _parse_page_spec(self, page_spec: str) -> List[int]:
        """
        Parse page specification like "1,2,10-20,55"

        Returns:
            List of page numbers
        """
        if not page_spec:
            return []

        pages = []
        parts = page_spec.split(',')

        for part in parts:
            part = part.strip()
            if '-' in part:
                # Range like "10-20"
                try:
                    start, end = part.split('-')
                    start_num = int(start.strip())
                    end_num = int(end.strip())
                    pages.extend(range(start_num, end_num + 1))
                except ValueError:
                    continue
            else:
                # Single page
                try:
                    pages.append(int(part))
                except ValueError:
                    continue

        return sorted(set(pages))

    async def get_visual_elements(
        self,
        pdf_id: str,
        page_numbers: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Get visual elements (images, diagrams, tables) for a PDF

        Args:
            pdf_id: PDF ID
            page_numbers: Optional list of specific page numbers to fetch

        Returns:
            List of visual element dicts [
                {
                    "id": int,
                    "pdf_id": str,
                    "page_number": int,
                    "type": str,  # "image", "table", "diagram", etc.
                    "path": str,  # path to image file
                    "bbox": str,  # bounding box JSON
                    "description": str  # text description from Gemini Vision
                },
                ...
            ]
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            if page_numbers:
                placeholders = ','.join('?' * len(page_numbers))
                sql = f'''
                    SELECT * FROM visual_elements
                    WHERE pdf_id = ? AND page_number IN ({placeholders})
                    ORDER BY page_number, id
                '''
                params = [pdf_id] + page_numbers
            else:
                sql = '''
                    SELECT * FROM visual_elements
                    WHERE pdf_id = ?
                    ORDER BY page_number, id
                '''
                params = [pdf_id]

            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


# Create singleton instance
pdf_database = PDFDatabase()

