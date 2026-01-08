"""
QA Service - Complete QA Flow
Ports the entire jsonResultsService.handleJsonResults() logic from Node.js
Maintains 100% compatibility with original QA flow
Enhanced with conversation context support
"""
import logging
import time
import json
import asyncio
import re
from typing import Dict, List, Optional, Any, Tuple

from app.services.llm_service import llm_service
from app.services.pdf_database import pdf_database
from app.services.qdrant_service import get_qdrant_service
from app.utils.conversation_cache import ConversationCache
from app.utils.llm_cache import get_llm_cache
from app.utils.conversation_logger import get_conversation_logger
from app.utils.query_logger import get_query_logger

logger = logging.getLogger(__name__)


class QAService:
    """
    Complete QA service that replicates Node.js jsonResultsService logic

    Flow:
    1. Get PDF metadata
    2. Get conversation history
    3. Call get_search_type() to determine search strategy
    4. Based on searchType, collect data (summary/toc/keyword/page)
    5. Call get_answer() with collected data
    6. Return results with metadata

    Enhanced with:
    - Conversation context support
    - LLM response caching
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Conversation cache for context persistence
        self.conversations = ConversationCache(max_size=1000, max_age_hours=24)
        # Context storage for followup questions
        self.context_store = {}
        # LLM response cache
        self.llm_cache = get_llm_cache()
        # Conversation file logger
        self.conversation_logger = get_conversation_logger()
        # Conversation summary storage (세션별 요약 관리)
        # {session_id: {"summary": str, "recent_turns": List[Dict]}}
        self.conversation_summaries = {}
        # 최근 대화 최대 개수
        self.MAX_RECENT_TURNS = 5

    async def handle_json_results(
        self,
        isbn: str,
        query: str,
        page_from: Optional[int],
        selected_text: Optional[str],
        userid: str,
        sessionid: str
    ) -> Dict[str, Any]:
        """
        Main QA handler - replicates jsonResultsService.handleJsonResults()

        Args:
            isbn: Book ISBN
            query: User query
            page_from: Optional starting page number
            selected_text: Optional selected text by user
            userid: User ID
            sessionid: Session ID

        Returns:
            Dict with infos, chunks, visualElements, contextInfo
        """
        # 🔍 Query Logger 초기화
        query_logger = get_query_logger()
        log_id = query_logger.start_log(
            userid=userid,
            sessionid=sessionid,
            isbn=isbn,
            query=query,
            select_text=selected_text
        )
        logger.info(f"📊 [QueryLog:{log_id}] Started tracking query: {query[:50]}")

        # Initialize variables for all code paths
        pages_spec = None
        pages_found = []
        search_type = None

        # 1) Get PDF metadata
        pdf = await pdf_database.get_pdf_by_isbn(isbn)
        if not pdf:
            raise ValueError(f"PDF with ISBN {isbn} not found")

        record = {
            'id': pdf['id'],
            'isbn': pdf['isbn'],
            'title': pdf.get('title', 'Unknown'),
            'file_path': pdf.get('file_path', '')
        }

        # 🚀 전체 요청 캐시 체크 (빠른 경로)
        # 단, 문맥 의존적인 질문은 캐싱하지 않음 (대화 흐름 반영 필요)
        context_dependent_keywords = [
            # 상세도 관련
            '더', '자세', '상세', '구체',
            # 연속성 관련
            '계속', '이어', '추가', '다시',
            # 접속사/지시어
            '그럼', '그러면', '그래서', '그런데', '하지만', '그리고',
            # 지시대명사
            '위', '아래', '앞', '뒤',
            '이거', '저거', '그거', '이것', '저것', '그것',
            '이전', '방금', '아까',
            # 짧은 대답/질문 (거의 항상 문맥 의존)
            '네', '응', '왜', '예를', '어떻게'
        ]

        # 매우 짧은 질문 (<= 3자)은 거의 항상 문맥 의존
        is_very_short = len(query.strip()) <= 3

        is_context_dependent = any(keyword in query for keyword in context_dependent_keywords) or is_very_short

        # 📊 Step 2: 대화 컨텍스트 로드
        conversation_context_formatted = await self._format_conversation_context(sessionid)
        previous_context = await self._get_latest_conversation_context(userid, sessionid)

        # 대화 기록 개수 계산
        conv_history = await self.conversations.get(sessionid)
        conv_turns = len(conv_history) if conv_history else 0

        query_logger.log_conversation_context(
            has_history=(conv_turns > 0),
            turns=conv_turns,
            context_dependent=is_context_dependent
        )

        # 📊 Step 3: 캐시 체크
        cache_start = time.time()
        if not is_context_dependent:
            # 독립적인 질문만 캐싱 (쿼리는 llm_cache에서 정규화됨)
            full_cache_context = f"full:{isbn}"
            cached_full_result = self.llm_cache.get(query, full_cache_context)
            cache_time_ms = (time.time() - cache_start) * 1000

            if cached_full_result:
                self.logger.info(f"⚡ FULL CACHE HIT - Instant response: {query[:50]}")
                query_logger.log_cache_check('full', True, cache_time_ms)

                # 캐시 히트 시 즉시 로그 저장하고 반환
                await query_logger.save_log(
                    answer=cached_full_result.get('answer', ''),
                    chunks_count=len(cached_full_result.get('chunks', [])),
                    visual_elements_count=len(cached_full_result.get('visualElements', []))
                )
                return cached_full_result
            else:
                query_logger.log_cache_check('full', False, cache_time_ms)
        else:
            self.logger.info(f"🔄 Context-dependent query, skipping cache: {query[:50]}")
            query_logger.log_cache_check('none', False, 0)

        # 📊 Step 4: TOC 로드
        toc_start = time.time()
        toc_core_summary = await self._get_toc_core_summary_text(record['id'])
        toc = toc_core_summary  # Use core_summary for LLM (includes keywords like MBTI)
        toc_time_ms = (time.time() - toc_start) * 1000

        query_logger.log_toc_load(
            toc_type='core_summary',
            toc_length=len(toc),
            load_time_ms=toc_time_ms
        )

        # Prepare user search
        user_search = (query or '').strip()
        if not user_search:
            user_search = "안녕하세요."

        # Handle direct title queries (bypass LLM to avoid hallucination)
        title_keywords = ['제목', '책 이름', '책이름', '타이틀', 'title', '이 책 이름', '도서명', '서명']
        query_lower = user_search.lower()
        if any(keyword in query_lower for keyword in title_keywords):
            # Check if this is a title-asking question (not just mentions title in passing)
            asking_patterns = ['뭐', '무엇', '알려', '이름', '?', '뭐야', '뭔가', '뭡니까']
            if any(pattern in query_lower for pattern in asking_patterns):
                self.logger.info(f'📚 Direct title query detected: {user_search}')
                title_answer = f"이 책의 제목은 **{record['title']}** 입니다."

                # Save to conversation cache
                await self.conversations.append(sessionid, [
                    {'role': 'user', 'content': user_search},
                    {'role': 'assistant', 'content': title_answer}
                ])

                # Return in the same format as normal responses
                infos = {
                    'userid': userid,
                    'sessionid': sessionid,
                    '제목': record['title'],
                    'isbn': record['isbn'],
                    '유저발화': user_search,
                    '검색유형': 'title_query',
                    '검색키워드': None,
                    '페이지범위': None,
                    '검색된페이지': [],
                    '검색선택이유': 'Direct title query detected',
                    '검색요청LLM시간': 0,
                    '데이터베이스검색시간': 0,
                    '답변생성시간': 0,
                    'answer': title_answer
                }

                return {
                    'infos': infos,
                    'chunks': [],
                    'visualElements': [],
                    'contextInfo': '{}'
                }

        # 📊 Step 5: LLM Call #1 - Determine search strategy
        # NOTE: irrelevant 판단을 위해 최근 대화 컨텍스트 제공 (후속 질문 감지)
        start_time = time.time()
        search_info_from_llm = await llm_service.get_search_type(
            title=record['title'],
            toc=toc,  # Use core_summary (includes keywords like MBTI)
            history_user_query=conversation_context_formatted,  # 후속 질문 감지를 위해 필요
            query=user_search,
            previous_context=previous_context
        )
        llm_time_ms = round((time.time() - start_time) * 1000, 1)

        search_result = search_info_from_llm['result']

        # 📊 Log LLM search type decision
        query_logger.log_llm_search_type(
            request_time_ms=llm_time_ms,
            search_type=search_result.get('searchType', 'unknown'),
            reason=search_result.get('reason', ''),
            core_keywords=search_result.get('coreKeywords', []),
            sub_keywords=search_result.get('subKeywords', []),
            page_range=search_result.get('pages'),
            use_previous_context=search_result.get('usePreviousContext', False)
        )

        # Handle previous context reuse for followup questions
        use_previous_context = search_result.get('usePreviousContext', False) and previous_context is not None
        if use_previous_context:
            self.logger.info(f'🔄 Using previous context for followup question')
            # Reuse search parameters from previous context
            search_result['searchType'] = previous_context.get('searchType', search_result.get('searchType'))
            search_result['pages'] = previous_context.get('pages', search_result.get('pages', ''))
            search_result['coreKeywords'] = previous_context.get('coreKeywords', search_result.get('coreKeywords', []))
            search_result['subKeywords'] = previous_context.get('subKeywords', search_result.get('subKeywords', []))

        # Extract page range from LLM result
        from_page = None
        to_page = None
        if 'startPage' in search_result:
            sp = search_result.get('startPage')
            ep = search_result.get('endPage')
            if sp and int(sp) > 0:
                from_page = int(sp)
            if ep and int(ep) > 0:
                to_page = int(ep)

            if from_page and not to_page:
                to_page = from_page
            if to_page and not from_page:
                from_page = to_page

        # 5) Collect data based on search type
        # NOTE: 항상 새로 검색 (previous_context에는 메타데이터만 있음)
        # usePreviousContext=true 이면 이전 검색 파라미터로 재검색
        book_texts = ""
        texts_summary = ""
        db_time_ms = 0
        pages_found = []
        search_logs = []
        chunks = []
        visual_elements = []

        core_keywords = search_result.get('coreKeywords', [])
        sub_keywords = search_result.get('subKeywords', [])
        search_type = search_result.get('searchType', 'keyword')
        pages_spec = search_result.get('pages', '')

        if use_previous_context:
            self.logger.info(f'🔄 Re-searching with previous metadata: {search_type}, pages={pages_spec}')

        # 항상 검색 수행 (메타데이터 기반)
        if True:  # 기존 조건문 제거
            if search_type == "irrelevant":
                # 책과 관련 없는 질문 - 검색 생략
                book_texts = ""
                texts_summary = ""
                pages_found = []
                self.logger.info(f'⚠️ Irrelevant question detected: {user_search}')

            elif search_type == "summary":
                # Get summary data from TOC
                texts_summary = await self._get_toc_summaries_by_page_spec(record['id'], pages_spec)
                # Parse pages_spec to populate pages_found for source citation (supports multiple ranges like "580-582,628,687-688")
                pages_found = self._parse_page_spec(pages_spec) if pages_spec else []

            elif search_type == "toc":
                # TOC query - no content needed
                book_texts = ""
                pages_found = []

            else:
                # keyword or page search
                has_keywords = len(core_keywords) > 0

                if not has_keywords:
                    # No keywords - get pages directly
                    if not pages_spec:
                        pages_spec = f"{from_page}-{to_page}" if from_page else "1-5"

                    start_time = time.time()
                    chunks = await self._get_chunks_text_by_page_spec(record['id'], pages_spec)
                    db_time_ms = round((time.time() - start_time) * 1000, 3)

                    if chunks:
                        pages_found = sorted(set(c['page'] for c in chunks))

                        if len(pages_found) <= 20:
                            book_texts = "\n\n".join([
                                f"page: {c['page']}\n\n내용: {c['content'].strip()}"
                                for c in chunks
                            ])
                        else:
                            book_texts = ""

                else:
                    # Keyword search with FTS
                    pages_search = self._parse_page_spec(pages_spec) if pages_spec else None

                    start_time = time.time()
                    result = await self._fetch_results_with_keywords(
                        pdf_id=record['id'],
                        core_keywords=core_keywords,
                        sub_keywords=sub_keywords,
                        pages=pages_search
                    )
                    db_time_ms = round((time.time() - start_time) * 1000, 3)

                    chunks = result['chunks']
                    visual_elements = result.get('visualElements', [])
                    search_logs = result.get('searchLogs', [])

                    # 📊 Log DB search (FTS5)
                    pages_found_list = sorted(set(c['page'] for c in chunks)) if chunks else []
                    query_logger.log_db_search(
                        method='fts5',
                        query=f"FTS5 MATCH with {len(core_keywords)} core + {len(sub_keywords)} sub keywords",
                        time_ms=db_time_ms,
                        results_count=len(chunks),
                        pages_found=pages_found_list,
                        extra_data={
                            'fts5': {
                                'keywords_used': {
                                    'core': [kw.get('keyword', '') for kw in core_keywords],
                                    'sub': [kw.get('keyword', '') for kw in sub_keywords]
                                },
                                'page_restriction': pages_spec,
                                'fallback_triggered': False
                            }
                        }
                    )

                    if chunks:
                        pages_found = sorted(set(c['page'] for c in chunks))

                        if len(pages_found) <= 20:
                            book_texts = "\n\n".join([
                                f"page: {c['page']}\n\n내용: {c['content'].strip()}"
                                for c in chunks
                            ])
                        else:
                            book_texts = ""

        # 6) LLM Call #2: Get answer (with caching)
        answer = ""
        answer_token = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        answer_time_ms = 0
        cache_hit = False

        if user_search:
            # Create cache context key (ISBN + search_type만 포함, 내용 제외)
            # 같은 질문이면 항상 같은 캐시 키 생성 → 캐시 히트율 향상
            cache_context = f"{isbn}:{search_type}"

            # 문맥 의존적인 질문은 LLM 답변 캐시도 스킵
            context_dependent_keywords = [
                # 상세도 관련
                '더', '자세', '상세', '구체',
                # 연속성 관련
                '계속', '이어', '추가', '다시',
                # 접속사/지시어
                '그럼', '그러면', '그래서', '그런데', '하지만', '그리고',
                # 지시대명사
                '위', '아래', '앞', '뒤',
                '이거', '저거', '그거', '이것', '저것', '그것',
                '이전', '방금', '아까',
                # 짧은 대답/질문
                '네', '응', '왜', '예를', '어떻게'
            ]
            is_very_short = len(user_search.strip()) <= 3
            is_context_dependent_query = any(keyword in user_search for keyword in context_dependent_keywords) or is_very_short

            # Check cache first (문맥 독립적인 질문만)
            if not is_context_dependent_query:
                cached_response = self.llm_cache.get(user_search, cache_context)
            else:
                cached_response = None
                self.logger.info(f"🔄 Context-dependent, skip LLM cache: {user_search[:50]}")

            if cached_response:
                # Cache hit!
                cache_hit = True
                llm_answer = cached_response
                answer_time_ms = 0  # Instant from cache
                self.logger.info(f"✅ LLM Cache HIT for: {user_search[:50]}")
            else:
                # Cache miss - call LLM
                start_time = time.time()

                if search_type == "summary":
                    llm_answer = await llm_service.get_answer(
                        title=record['title'],
                        texts=texts_summary,
                        history_user_query=conversation_context_formatted,
                        question=user_search,
                        toc=""
                    )
                elif search_type == "toc":
                    # Determine detail level based on query keywords
                    # Show full details if user asks for "전체", "전부", "모두", "상세", "다 보여"
                    query_lower = user_search.lower()
                    show_full_details = any(keyword in query_lower for keyword in ['전체', '전부', '모두', '상세', '다 보여', '세부'])

                    # Always use full TOC data, let LLM filter based on prompt instruction
                    llm_answer = await llm_service.get_toc_answer(
                        title=record['title'],
                        toc=toc,  # Use core_summary (includes keywords)
                        question=user_search,
                        show_full_details=show_full_details,
                        history_user_query=conversation_context_formatted
                    )
                else:
                    llm_answer = await llm_service.get_answer(
                        title=record['title'],
                        texts=book_texts,
                        history_user_query=conversation_context_formatted,
                        question=user_search,
                        toc=toc  # Use core_summary (includes keywords)
                    )

                answer_time_ms = round((time.time() - start_time) * 1000, 1)

                # 📊 Log LLM answer generation
                query_logger.log_llm_answer(
                    request_time_ms=answer_time_ms,
                    prompt_length=len(book_texts) if book_texts else 0,
                    chunks_count=len(chunks) if chunks else 0,
                    answer_length=len(llm_answer.get('result', ''))
                )

                # Save to cache (문맥 독립적인 질문만)
                if not is_context_dependent_query:
                    self.llm_cache.set(user_search, cache_context, llm_answer)
                    self.logger.info(f"💾 LLM Cache SET for: {user_search[:50]}")
                else:
                    self.logger.info(f"🔄 Context-dependent, LLM answer not cached: {user_search[:50]}")

            answer = llm_answer['result']
            answer_token = llm_answer['token']

        # 자동 출처 생성 (LLM이 아닌 시스템에서)
        source_citation = await self._generate_source_citation(
            pdf_id=record['id'],
            pages_found=pages_found,
            search_type=search_type
        )
        if source_citation:
            answer = answer + source_citation

        # Fetch visual elements (images, diagrams, tables) for found pages
        if pages_found:
            try:
                visual_elements = await pdf_database.get_visual_elements(
                    pdf_id=record['id'],
                    page_numbers=pages_found
                )
                self.logger.info(f"📊 Found {len(visual_elements)} visual elements on pages {pages_found[:3]}...")
            except Exception as e:
                self.logger.error(f"❌ Failed to fetch visual elements: {e}")
                visual_elements = []

        # 7) Build infos object (flat structure like Node.js)
        search_token = search_info_from_llm['token']
        core_keyword_str = ' '.join([f'"{k["keyword"]}"' for k in core_keywords]) if core_keywords else ""
        sub_keyword_str = ' '.join([f'"{k["keyword"]}"' for k in sub_keywords]) if sub_keywords else ""

        keyword_str = ""
        if core_keyword_str or sub_keyword_str:
            keyword_str = f"핵심: {core_keyword_str}"
            if sub_keyword_str:
                keyword_str += f" / 보조: {sub_keyword_str}"

        infos = {
            '답변전체_토큰': f"input :{search_token.get('prompt_tokens', 0) + answer_token.get('prompt_tokens', 0)} output : {search_token.get('completion_tokens', 0) + answer_token.get('completion_tokens', 0)}",
            'userid': userid,
            'sessionid': sessionid,
            '제목': record['title'],
            'isbn': record['isbn'],
            '사용자정보': "",
            '유저발화': query,
            'nowPage': "",
            'nowToc': "",
            'selectText': selected_text or None,

            '타입검색결과': str(search_info_from_llm),
            '검색유형': search_result.get('searchType'),
            '검색키워드': keyword_str if keyword_str else None,
            '페이지범위': pages_spec or None,
            '검색된페이지': pages_found if pages_found else [],
            '시작페이지': search_result.get('startPage'),
            '끝페이지': search_result.get('endPage'),

            '검색선택이유': search_result.get('reason'),
            '검색로그': str(search_logs) if search_logs else None,
            '검색생성사용토큰정보': search_token,
            '검색생성토큰': search_token.get('prompt_tokens', 0),
            '검색생성아웃풋토큰': search_token.get('completion_tokens', 0),
            '검색요청LLM시간': llm_time_ms,

            '데이터베이스검색시간': db_time_ms,

            '답변생성시간': answer_time_ms,
            '답변생성사용토큰': answer_token,
            '답변생성인풋토큰': answer_token.get('prompt_tokens', 0),
            '답변생성아웃풋토큰': answer_token.get('completion_tokens', 0),
            'answer': answer or None,
        }

        # Context info for next turn
        # NOTE: texts/chunks 등 원문 데이터는 저장하지 않음 (프롬프트 길이 폭발 방지)
        # 필요시 메타데이터로 재검색
        context_info = {
            'searchType': search_result.get('searchType'),
            'pages': pages_spec,
            'coreKeywords': core_keywords,
            'subKeywords': sub_keywords,
            'pagesFound': pages_found,
            'query': query  # 직전 사용자 질문 저장
            # 'texts': book_texts,  # ❌ 제거 - 재검색으로 대체
            # 'texts_summary': texts_summary,  # ❌ 제거
            # 'chunks': chunks,  # ❌ 제거
            # 'visualElements': visual_elements  # ❌ 제거
        }

        # Save context for followup questions
        context_key = f"{userid}:{sessionid}"
        self.context_store[context_key] = context_info
        self.logger.info(f'💾 Saved context (metadata only) for session: {sessionid}')

        # Save conversation history to cache
        await self.conversations.append(sessionid, [
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': answer}
        ])
        self.logger.info(f'💬 Saved conversation to cache: {sessionid}')

        # Save conversation to file
        await self.conversation_logger.log_conversation(
            session_id=sessionid,
            user_id=userid,
            isbn=isbn,
            query=query,
            response=answer,
            metadata={
                'search_type': search_type,
                'pages': pages_spec,
                'pages_found': len(chunks) if chunks else 0,
                'response_length': len(answer)
            }
        )

        # Update conversation summary + recent turns (백그라운드 실행 - 응답 블록 안 함)
        asyncio.create_task(self._update_conversation_summary(
            sessionid=sessionid,
            user_query=query,
            assistant_response=answer,
            isbn=isbn
        ))

        # 🚀 전체 요청 결과를 캐시에 저장 (문맥 독립적인 질문만)
        result = {
            'infos': infos,
            'chunks': chunks,
            'visualElements': visual_elements,
            'contextInfo': str(context_info)
        }

        # 문맥 의존적인 질문은 캐싱하지 않음
        context_dependent_keywords = [
            # 상세도 관련
            '더', '자세', '상세', '구체',
            # 연속성 관련
            '계속', '이어', '추가', '다시',
            # 접속사/지시어
            '그럼', '그러면', '그래서', '그런데', '하지만', '그리고',
            # 지시대명사
            '위', '아래', '앞', '뒤',
            '이거', '저거', '그거', '이것', '저것', '그것',
            '이전', '방금', '아까',
            # 짧은 대답/질문
            '네', '응', '왜', '예를', '어떻게'
        ]
        is_very_short = len(query.strip()) <= 3
        is_context_dependent = any(keyword in query for keyword in context_dependent_keywords) or is_very_short

        if not is_context_dependent:
            full_cache_context = f"full:{isbn}"
            self.llm_cache.set(query, full_cache_context, result)
            self.logger.info(f"💾 FULL RESULT CACHED: {query[:50]}")
        else:
            self.logger.info(f"🔄 Context-dependent query, not cached: {query[:50]}")

        # 📊 Save query log to DB
        try:
            await query_logger.save_log(
                answer=answer or '',
                chunks_count=len(chunks),
                visual_elements_count=len(visual_elements)
            )
            self.logger.info(f"📊 [QueryLog:{log_id}] Saved to DB")
        except Exception as e:
            self.logger.error(f"❌ Failed to save query log: {e}")

        return result

    async def handle_json_results_stream(
        self,
        isbn: str,
        query: str,
        page_from: Optional[int],
        selected_text: Optional[str],
        userid: str,
        sessionid: str
    ):
        """
        Streaming version of handle_json_results

        Yields chunks of the answer in real-time as they're generated.
        Note: Streaming bypasses LLM cache for real-time delivery.

        Yields:
            Dict with:
            - chunk: Text chunk (or "" for metadata)
            - done: Boolean indicating completion
            - metadata: (Only on final message) Full response metadata
        """
        try:
            # ⏱️ 시작 시간 기록
            start_time = time.time()

            # 🔍 Query Logger 초기화 (streaming)
            query_logger = get_query_logger()
            log_id = query_logger.start_log(
                userid=userid,
                sessionid=sessionid,
                isbn=isbn,
                query=query,
                select_text=selected_text
            )
            self.logger.info(f"📊 [QueryLog:{log_id}] Started tracking query (streaming): {query[:50]}")

            # 1) Get PDF metadata
            pdf = await pdf_database.get_pdf_by_isbn(isbn)
            if not pdf:
                raise ValueError(f"PDF with ISBN {isbn} not found")

            record = {
                'id': pdf['id'],
                'isbn': pdf['isbn'],
                'title': pdf.get('title', 'Unknown'),
                'file_path': pdf.get('file_path', '')
            }

            # 🚀 전체 요청 캐시 체크 (스트리밍 API도 캐싱 지원!)
            # 문맥 의존적인 질문은 캐싱하지 않음
            user_search = (query or '').strip()
            if not user_search:
                user_search = "안녕하세요."

            self.logger.info(f"🔍 [STREAM] Checking cache for: {user_search[:50]}")

            context_dependent_keywords = [
                '더', '자세', '상세', '구체',
                '계속', '이어', '추가', '다시',
                '그럼', '그러면', '그래서', '그런데', '하지만', '그리고',
                '위', '아래', '앞', '뒤',
                '이거', '저거', '그거', '이것', '저것', '그것',
                '이전', '방금', '아까',
                '네', '응', '왜', '예를', '어떻게'
            ]

            is_very_short = len(user_search.strip()) <= 3
            is_context_dependent = any(keyword in user_search for keyword in context_dependent_keywords) or is_very_short

            if not is_context_dependent:
                # 독립적인 질문만 캐싱
                cache_start_time = time.time()
                full_cache_context = f"full:{isbn}"
                cached_full_result = self.llm_cache.get(user_search, full_cache_context)
                cache_hit_time_ms = round((time.time() - cache_start_time) * 1000, 3)

                # 📊 로깅: Cache Check
                query_logger.log_cache_check(
                    cache_type="full",
                    hit=cached_full_result is not None,
                    hit_time_ms=cache_hit_time_ms
                )

                if cached_full_result:
                    self.logger.info(f"⚡ STREAM CACHE HIT - Instant response: {user_search[:50]}")

                    # 캐시된 응답을 청크로 나눠서 스트리밍
                    answer = cached_full_result['answer']
                    chunk_size = 10  # 10자씩 청크
                    for i in range(0, len(answer), chunk_size):
                        chunk = answer[i:i+chunk_size]
                        yield {
                            'chunk': chunk,
                            'done': False
                        }

                    # 메타데이터와 함께 완료 신호
                    yield {
                        'chunk': '',
                        'done': True,
                        'metadata': cached_full_result
                    }
                    return
                else:
                    self.logger.debug(f"📊 [QueryLog] Cache miss: {user_search[:50]}")
            else:
                self.logger.info(f"🔄 Context-dependent query (stream), skipping cache: {user_search[:50]}")

            # 2) Get conversation context (summary + recent turns)
            # 기존 전체 히스토리 대신 요약 + 최근 N턴만 사용 (프롬프트 길이 고정)
            conversation_context_formatted = await self._format_conversation_context(sessionid)

            previous_context = await self._get_latest_conversation_context(userid, sessionid)

            # 📊 로깅: Conversation Context
            conv_history = await self.conversations.get(sessionid)
            conv_turns = len(conv_history) if conv_history else 0
            query_logger.log_conversation_context(
                has_history=conv_turns > 0,
                turns=conv_turns,
                context_dependent=is_context_dependent
            )
            self.logger.debug(f"📊 [QueryLog] Logged conversation context: {conv_turns} turns")

            # 2.1) Check if this is a new session - create session without greeting
            is_new_session = sessionid not in self.conversation_summaries or not self.conversation_summaries[sessionid]["recent_turns"]
            if is_new_session:
                self.logger.info(f"✨ New session detected: {sessionid}")

                # Create session silently (no greeting message)
                await self.conversation_logger.create_session(
                    session_id=sessionid,
                    user_id=userid,
                    isbn=isbn,
                    system_message=""
                )

            # 3) Get TOC (Table of Contents) with core_summary for better keyword matching
            toc_start_time = time.time()
            toc_core_summary = await self._get_toc_core_summary_text(record['id'])
            toc = toc_core_summary  # Use core_summary for LLM (includes keywords like MBTI)
            toc_load_time_ms = round((time.time() - toc_start_time) * 1000, 3)

            # 📊 로깅: TOC Load
            query_logger.log_toc_load(
                toc_type="core_summary",
                toc_length=len(toc) if toc else 0,
                load_time_ms=toc_load_time_ms
            )
            self.logger.debug(f"📊 [QueryLog] Loaded TOC: {len(toc) if toc else 0} chars in {toc_load_time_ms}ms")

            # user_search는 위에서 이미 정의됨 (캐시 체크 부분)

            # Handle direct title queries (bypass LLM to avoid hallucination)
            title_keywords = ['제목', '책 이름', '책이름', '타이틀', 'title', '이 책 이름', '도서명', '서명']
            query_lower = user_search.lower()
            if any(keyword in query_lower for keyword in title_keywords):
                # Check if this is a title-asking question (not just mentions title in passing)
                asking_patterns = ['뭐', '무엇', '알려', '이름', '?', '뭐야', '뭔가', '뭡니까']
                if any(pattern in query_lower for pattern in asking_patterns):
                    self.logger.info(f'📚 Direct title query detected (streaming): {user_search}')
                    title_answer = f"이 책의 제목은 **{record['title']}** 입니다."

                    # Save to conversation cache
                    await self.conversations.append(sessionid, [
                        {'role': 'user', 'content': user_search},
                        {'role': 'assistant', 'content': title_answer}
                    ])

                    # Return in the same format as normal streaming responses
                    infos = {
                        'userid': userid,
                        'sessionid': sessionid,
                        '제목': record['title'],
                        'isbn': record['isbn'],
                        '유저발화': user_search,
                        '검색유형': 'title_query',
                        '검색키워드': None,
                        '페이지범위': None,
                        '검색된페이지': [],
                        '검색선택이유': 'Direct title query detected',
                        '검색요청LLM시간': 0,
                        '데이터베이스검색시간': 0,
                        '답변생성시간': 0,
                        'answer': title_answer
                    }

                    # Yield the answer as a stream
                    yield {
                        'chunk': title_answer,
                        'done': False
                    }
                    yield {
                        'chunk': '',
                        'done': True,
                        'metadata': {
                            'infos': infos,
                            'chunks': [],
                            'visualElements': [],
                            'contextInfo': '{}'
                        }
                    }
                    return

            # 3.5) 🔥 Smart chapter detection for large TOCs
            # Pattern: "N장 요약", "제N장", "Chapter N" etc.
            chapter_match = re.search(r'(?:제?\s*)?(\d+)\s*장', user_search)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                self.logger.info(f"🔍 [Chapter Detection] Detected chapter {chapter_num} request")

                # Try to find chapter in TOC directly from database
                try:
                    import aiosqlite
                    async with aiosqlite.connect(pdf_database.db_path) as db:
                        # Get all entries that start with "N장" (main chapter + all subsections)
                        # Match both "N장 ..." and "N-1 ...", "N-2-1 ..." patterns
                        cursor = await db.execute(
                            """
                            SELECT title, start_page, end_page
                            FROM toc_entries
                            WHERE pdf_id = ? AND (title LIKE ? OR title LIKE ?)
                            ORDER BY start_page ASC
                            """,
                            (record['id'], f'{chapter_num}장%', f'{chapter_num}-%')
                        )
                        rows = await cursor.fetchall()

                        if rows:
                            # Get main chapter title from first entry
                            chapter_title = rows[0][0]
                            # Get start page from first entry, end page from last entry
                            start_page = rows[0][1]
                            end_page = rows[-1][2]
                            pages_spec = f"{start_page}-{end_page}"

                            self.logger.info(f"✅ [Chapter Detection] Found: {chapter_title} (pages {pages_spec})")
                            self.logger.info(f"   - Includes {len(rows)} sections from '{rows[0][0]}' to '{rows[-1][0]}'")

                            # Override search_result to use direct chapter lookup
                            search_result = {
                                'searchType': 'summary',
                                'pages': pages_spec,
                                'coreKeywords': [],
                                'subKeywords': [],
                                'usePreviousContext': False,
                                'reason': f'Chapter {chapter_num} detected via pattern matching ({len(rows)} sections, {start_page}-{end_page})'
                            }
                            llm_time_ms = 0.0  # Skip LLM call

                            # Skip LLM search type call
                            goto_search_result = True
                            chapter_detected_logged = True  # Flag to prevent duplicate logging
                        else:
                            self.logger.warning(f"⚠️ [Chapter Detection] Chapter {chapter_num} not found in TOC")
                            goto_search_result = False
                            chapter_detected_logged = False
                except Exception as e:
                    self.logger.error(f"❌ [Chapter Detection] Error: {e}")
                    goto_search_result = False
                    chapter_detected_logged = False
            else:
                goto_search_result = False
                chapter_detected_logged = False

            if not goto_search_result:
                # 4) LLM Call #1: Determine search strategy (non-streaming)
                # NOTE: irrelevant 판단을 위해 최근 대화 컨텍스트 제공
                llm_search_start_time = time.time()
                search_info_from_llm = await llm_service.get_search_type(
                    title=record['title'],
                    toc=toc,  # Use core_summary (includes keywords like MBTI)
                    history_user_query=conversation_context_formatted,  # 후속 질문 감지
                    query=user_search,
                    previous_context=previous_context
                )
                llm_time_ms = round((time.time() - llm_search_start_time) * 1000, 1)
                search_result = search_info_from_llm['result']
            else:
                # Skip LLM call, use detected chapter info
                search_info_from_llm = {'result': search_result, 'token': {}}

            # 📊 로깅: LLM Search Type Decision (if not already logged)
            if not chapter_detected_logged:
                query_logger.log_llm_search_type(
                    request_time_ms=llm_time_ms,
                    search_type=search_result.get('searchType', 'keyword'),
                    reason=search_result.get('reason', ''),
                core_keywords=search_result.get('coreKeywords', []),
                sub_keywords=search_result.get('subKeywords', []),
                page_range=search_result.get('pages', ''),
                use_previous_context=search_result.get('usePreviousContext', False)
            )
            self.logger.debug(f"📊 [QueryLog] LLM decided: {search_result.get('searchType')} in {llm_time_ms}ms")

            # Handle previous context reuse
            use_previous_context = search_result.get('usePreviousContext', False) and previous_context is not None
            if use_previous_context:
                self.logger.info(f'🔄 Using previous context for followup question')
                search_result['searchType'] = previous_context.get('searchType', search_result.get('searchType'))
                search_result['pages'] = previous_context.get('pages', search_result.get('pages', ''))
                search_result['coreKeywords'] = previous_context.get('coreKeywords', search_result.get('coreKeywords', []))
                search_result['subKeywords'] = previous_context.get('subKeywords', search_result.get('subKeywords', []))

            # Extract page range
            from_page = None
            to_page = None
            if 'startPage' in search_result:
                sp = search_result.get('startPage')
                ep = search_result.get('endPage')
                if sp and int(sp) > 0:
                    from_page = int(sp)
                if ep and int(ep) > 0:
                    to_page = int(ep)

                if from_page and not to_page:
                    to_page = from_page
                if to_page and not from_page:
                    from_page = to_page

            # 5) Collect data based on search type
            # NOTE: 항상 새로 검색 (previous_context에는 메타데이터만 있음)
            # usePreviousContext=true 이면 이전 검색 파라미터로 재검색
            book_texts = ""
            texts_summary = ""
            db_time_ms = 0
            pages_found = []
            search_logs = []
            chunks = []
            visual_elements = []

            core_keywords = search_result.get('coreKeywords', [])
            sub_keywords = search_result.get('subKeywords', [])
            search_type = search_result.get('searchType', 'keyword')
            pages_spec = search_result.get('pages', '')

            if use_previous_context:
                self.logger.info(f'🔄 Re-searching with previous metadata: {search_type}, pages={pages_spec}')

            # 항상 검색 수행 (메타데이터 기반)
            if True:  # 기존 조건문 제거
                if search_type == "irrelevant":
                    # 책과 관련 없는 질문 - 검색 생략
                    book_texts = ""
                    texts_summary = ""
                    pages_found = []
                    self.logger.info(f'⚠️ Irrelevant question detected (stream): {user_search}')

                elif search_type == "summary":
                    texts_summary = await self._get_toc_summaries_by_page_spec(record['id'], pages_spec)
                    # Parse pages_spec to populate pages_found for source citation (supports multiple ranges like "580-582,628,687-688")
                    pages_found = self._parse_page_spec(pages_spec) if pages_spec else []

                elif search_type == "toc":
                    book_texts = ""
                    pages_found = []

                elif search_type == "semantic":
                    # Qdrant 시맨틱 검색
                    db_search_start = time.time()
                    try:
                        qdrant = get_qdrant_service()
                        semantic_results = await qdrant.search(
                            pdf_id=record['id'],
                            query=user_search,
                            limit=5
                        )
                        db_time_ms = round((time.time() - db_search_start) * 1000, 3)

                        if semantic_results:
                            chunks = []
                            for result in semantic_results:
                                chunks.append({
                                    'page': result['page_number'],
                                    'content': result['content']
                                })
                            pages_found = sorted(set(c['page'] for c in chunks))

                            if len(pages_found) <= 20:
                                book_texts = "\n\n".join([
                                    f"page: {c['page']}\n\n내용: {c['content'].strip()[:800]}"
                                    for c in chunks[:5]
                                ])
                            else:
                                book_texts = ""
                        else:
                            book_texts = ""
                            pages_found = []

                        self.logger.info(f"🔍 Semantic search found {len(chunks)} chunks")

                        # 📊 로깅: DB Search (Qdrant)
                        query_logger.log_db_search(
                            method="qdrant",
                            query=user_search,
                            time_ms=db_time_ms,
                            results_count=len(chunks),
                            pages_found=pages_found,
                            extra_data={
                                "qdrant_top_k": 5,
                                "qdrant_fallback_to_keyword": False
                            }
                        )
                        self.logger.debug(f"📊 [QueryLog] Qdrant search: {len(chunks)} results in {db_time_ms}ms")

                    except Exception as e:
                        self.logger.warning(f"⚠️ Semantic search failed, falling back to keyword: {e}")

                        # 📊 로깅: Qdrant Fallback
                        query_logger.log_db_search(
                            method="qdrant",
                            query=user_search,
                            time_ms=db_time_ms if 'db_time_ms' in locals() else 0,
                            results_count=0,
                            pages_found=[],
                            extra_data={
                                "qdrant_fallback_to_keyword": True,
                                "error": str(e)
                            }
                        )

                        search_type = "keyword"  # Fallback to keyword search
                        # Continue to keyword search below

                else:
                    has_keywords = len(core_keywords) > 0

                    if not has_keywords:
                        if not pages_spec:
                            pages_spec = f"{from_page}-{to_page}" if from_page else "1-5"

                        db_search_start = time.time()
                        chunks = await self._get_chunks_text_by_page_spec(record['id'], pages_spec)
                        db_time_ms = round((time.time() - db_search_start) * 1000, 3)

                        if chunks:
                            pages_found = sorted(set(c['page'] for c in chunks))

                            # 📊 로깅: DB Search (Page Spec)
                            query_logger.log_db_search(
                                method="page_spec",
                                query=f"pages:{pages_spec}",
                                time_ms=db_time_ms,
                                results_count=len(chunks),
                                pages_found=pages_found
                            )
                            self.logger.debug(f"📊 [QueryLog] Page spec search: {len(chunks)} results from pages {pages_spec}")

                            if len(pages_found) <= 20:
                                # LOW EFFORT: 각 청크 내용을 800자로 제한 (프롬프트 길이 감소)
                                book_texts = "\n\n".join([
                                    f"page: {c['page']}\n\n내용: {c['content'].strip()[:800]}"
                                    for c in chunks[:5]  # 최대 5개 청크만
                                ])
                            else:
                                book_texts = ""

                    else:
                        pages_search = self._parse_page_spec(pages_spec) if pages_spec else None

                        db_search_start = time.time()
                        result = await self._fetch_results_with_keywords(
                            pdf_id=record['id'],
                            core_keywords=core_keywords,
                            sub_keywords=sub_keywords,
                            pages=pages_search
                        )
                        db_time_ms = round((time.time() - db_search_start) * 1000, 3)

                        chunks = result['chunks']
                        visual_elements = result.get('visualElements', [])
                        search_logs = result.get('searchLogs', [])

                        fallback_triggered = False

                        # 📌 Fallback: If no results found with page restriction, retry without pages
                        if not chunks and pages_search:
                            self.logger.info(f"🔄 No results in pages {pages_spec}, retrying全 all pages...")
                            fallback_start = time.time()
                            result = await self._fetch_results_with_keywords(
                                pdf_id=record['id'],
                                core_keywords=core_keywords,
                                sub_keywords=sub_keywords,
                                pages=None  # Search ALL pages
                            )
                            chunks = result['chunks']
                            visual_elements = result.get('visualElements', [])
                            db_time_ms += round((time.time() - fallback_start) * 1000, 3)
                            fallback_triggered = True
                            if chunks:
                                self.logger.info(f"✅ Fallback found {len(chunks)} chunks without page restriction")

                        if chunks:
                            pages_found = sorted(set(c['page'] for c in chunks))

                            # 📊 로깅: DB Search (FTS5)
                            core_kw_list = [k['keyword'] for k in core_keywords]
                            sub_kw_list = [k['keyword'] for k in sub_keywords]
                            query_logger.log_db_search(
                                method="fts5",
                                query=f"core:{core_kw_list} sub:{sub_kw_list}",
                                time_ms=db_time_ms,
                                results_count=len(chunks),
                                pages_found=pages_found,
                                extra_data={
                                    "fts5_keywords_used": {
                                        "core": core_kw_list,
                                        "sub": sub_kw_list
                                    },
                                    "fts5_page_restriction": pages_spec if pages_spec else "none",
                                    "fts5_fallback_triggered": fallback_triggered
                                }
                            )
                            self.logger.debug(f"📊 [QueryLog] FTS5 search: {len(chunks)} results, fallback={fallback_triggered}")

                            if len(pages_found) <= 20:
                                # LOW EFFORT: 각 청크 내용을 800자로 제한 (프롬프트 길이 감소)
                                book_texts = "\n\n".join([
                                    f"page: {c['page']}\n\n내용: {c['content'].strip()[:800]}"
                                    for c in chunks[:5]  # 최대 5개 청크만
                                ])
                            else:
                                book_texts = ""

            # 6) LLM Call #2: Stream answer (NO caching)
            answer_chunks = []
            answer_time_start = time.time()

            if user_search:
                self.logger.info(f"🔄 Streaming answer for: {user_search[:50]}")

                # Stream the answer based on search type
                if search_type == "irrelevant":
                    # 책과 관련 없는 질문 - 안내 메시지 스트리밍
                    async def irrelevant_stream():
                        message = f"죄송합니다. 이 질문은 현재 선택하신 교재 '{record['title']}'의 내용과 관련이 없어 답변드릴 수 없습니다."
                        yield message
                    stream = irrelevant_stream()
                elif search_type == "summary":
                    stream = llm_service.get_answer_stream(
                        title=record['title'],
                        texts=texts_summary,
                        history_user_query=conversation_context_formatted,
                        question=user_search,
                        toc=""
                    )
                elif search_type == "toc":
                    stream = llm_service.get_answer_stream(
                        title=record['title'],
                        texts="",
                        history_user_query=conversation_context_formatted,
                        question=user_search,
                        toc=toc  # Use core_summary (includes keywords)
                    )
                else:
                    stream = llm_service.get_answer_stream(
                        title=record['title'],
                        texts=book_texts,
                        history_user_query=conversation_context_formatted,
                        question=user_search,
                        toc=toc  # Use core_summary (includes keywords)
                    )

                # Stream chunks to client
                async for chunk in stream:
                    answer_chunks.append(chunk)
                    yield {
                        'chunk': chunk,
                        'done': False
                    }

            answer_time_ms = round((time.time() - answer_time_start) * 1000, 1)
            answer = ''.join(answer_chunks)

            # 📊 로깅: LLM Answer Generation
            prompt_length = len(book_texts) + len(toc) + len(conversation_context_formatted) + len(user_search)
            query_logger.log_llm_answer(
                request_time_ms=answer_time_ms,
                prompt_length=prompt_length,
                chunks_count=len(chunks),
                answer_length=len(answer)
            )
            self.logger.debug(f"📊 [QueryLog] LLM answer: {len(answer)} chars in {answer_time_ms}ms")

            # 자동 출처 생성 (LLM이 아닌 시스템에서)
            source_citation = await self._generate_source_citation(
                pdf_id=record['id'],
                pages_found=pages_found,
                search_type=search_type
            )
            if source_citation:
                # ✅ Stream the source citation to client
                yield {
                    'chunk': source_citation,
                    'done': False
                }
                answer = answer + source_citation

            # Fetch visual elements (images, diagrams, tables) for found pages
            if pages_found:
                try:
                    visual_elements = await pdf_database.get_visual_elements(
                        pdf_id=record['id'],
                        page_numbers=pages_found
                    )
                    self.logger.info(f"📊 Found {len(visual_elements)} visual elements on pages {pages_found[:3]}...")
                except Exception as e:
                    self.logger.error(f"❌ Failed to fetch visual elements: {e}")
                    visual_elements = []

            # 7) Build metadata
            search_token = search_info_from_llm['token']
            core_keyword_str = ' '.join([f'"{k["keyword"]}"' for k in core_keywords]) if core_keywords else ""
            sub_keyword_str = ' '.join([f'"{k["keyword"]}"' for k in sub_keywords]) if sub_keywords else ""

            keyword_str = ""
            if core_keyword_str or sub_keyword_str:
                keyword_str = f"핵심: {core_keyword_str}"
                if sub_keyword_str:
                    keyword_str += f" / 보조: {sub_keyword_str}"

            infos = {
                'userid': userid,
                'sessionid': sessionid,
                '제목': record['title'],
                'isbn': record['isbn'],
                '유저발화': query,
                'selectText': selected_text or None,
                '검색유형': search_result.get('searchType'),
                '검색키워드': keyword_str if keyword_str else None,
                '페이지범위': pages_spec or None,
                '검색된페이지': pages_found if pages_found else [],
                '검색선택이유': search_result.get('reason'),
                '검색요청LLM시간': llm_time_ms,
                '데이터베이스검색시간': db_time_ms,
                '답변생성시간': answer_time_ms,
                'answer': answer or None,
            }

            # Save context for followup questions
            # NOTE: texts/chunks 등 원문 데이터는 저장하지 않음 (프롬프트 길이 폭발 방지)
            # 필요시 메타데이터로 재검색
            context_info = {
                'searchType': search_result.get('searchType'),
                'pages': pages_spec,
                'coreKeywords': core_keywords,
                'subKeywords': sub_keywords,
                'pagesFound': pages_found,
                'query': query  # 직전 사용자 질문 저장
                # 'texts': book_texts,  # ❌ 제거 - 재검색으로 대체
                # 'texts_summary': texts_summary,  # ❌ 제거
                # 'chunks': chunks,  # ❌ 제거
                # 'visualElements': visual_elements  # ❌ 제거
            }

            context_key = f"{userid}:{sessionid}"
            self.context_store[context_key] = context_info
            self.logger.info(f'💾 Saved context (metadata only) for session: {sessionid}')

            # Save conversation history
            await self.conversations.append(sessionid, [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': answer}
            ])
            self.logger.info(f'💬 Saved conversation to cache: {sessionid}')

            # Save conversation to file
            await self.conversation_logger.log_conversation(
                session_id=sessionid,
                user_id=userid,
                isbn=isbn,
                query=query,
                response=answer,
                metadata={
                    'search_type': search_result.get('searchType'),
                    'pages': pages_spec,
                    'pages_found': pages_found,
                    'response_length': len(answer),
                    'streaming': True
                }
            )

            # Update conversation summary + recent turns (백그라운드 실행 - 응답 블록 안 함)
            asyncio.create_task(self._update_conversation_summary(
                sessionid=sessionid,
                user_query=query,
                assistant_response=answer,
                isbn=isbn
            ))

            # 💾 스트리밍 응답도 캐시에 저장 (문맥 독립적인 질문만)
            if not is_context_dependent:
                full_cache_context = f"full:{isbn}"
                cache_data = {
                    'answer': answer,
                    'infos': infos,
                    'chunks': chunks,
                    'visualElements': visual_elements
                }
                self.llm_cache.set(user_search, full_cache_context, cache_data)
                self.logger.info(f"💾 Cached streaming response: {user_search[:50]}")

            # 📊 Save query log to DB (streaming)
            try:
                await query_logger.save_log(
                    answer=answer or '',
                    chunks_count=len(chunks),
                    visual_elements_count=len(visual_elements)
                )
                self.logger.info(f"📊 [QueryLog] Saved streaming log to DB")
            except Exception as log_error:
                self.logger.error(f"❌ Failed to save query log: {log_error}")

            # ⏱️ 총 소요 시간 계산
            total_time_ms = (time.time() - start_time) * 1000

            # Final message with metadata
            yield {
                'chunk': '',
                'done': True,
                'metadata': {
                    'infos': infos,
                    'chunks': chunks,
                    'visualElements': visual_elements,
                    'total_time_ms': round(total_time_ms, 2)
                }
            }

        except Exception as e:
            self.logger.error(f"Error in streaming handler: {e}", exc_info=True)
            yield {
                'chunk': '',
                'done': True,
                'error': str(e)
            }

    # ========== Helper Methods ==========

    async def _generate_source_citation(self, pdf_id: str, pages_found: List[int], search_type: str) -> str:
        """
        자동으로 출처를 생성합니다 (LLM이 아닌 시스템에서 생성)

        Args:
            pdf_id: PDF ID
            pages_found: 검색된 페이지 번호 리스트
            search_type: 검색 타입

        Returns:
            형식화된 출처 문자열 (예: "1-1 주제어 살펴보기 (19-26페이지)")
        """
        if not pages_found or search_type in ["irrelevant", "followup"]:
            return ""

        try:
            # 목차 정보 가져오기
            toc_entries = await pdf_database.get_toc_entries(pdf_id)
            if not toc_entries:
                return ""

            # 페이지를 목차 항목에 매핑
            page_to_toc = {}
            for entry in toc_entries:
                start = entry.get('start_page')
                end = entry.get('end_page')
                title = entry.get('title', '')

                if start and end:
                    for page in range(start, end + 1):
                        if page not in page_to_toc:
                            page_to_toc[page] = title

            # 검색된 페이지들을 목차 항목별로 그룹화
            toc_groups = {}
            for page in sorted(pages_found):
                toc_title = page_to_toc.get(page)
                if toc_title:
                    if toc_title not in toc_groups:
                        toc_groups[toc_title] = []
                    toc_groups[toc_title].append(page)

            if not toc_groups:
                return ""

            # 출처 문자열 생성
            sources = []
            for toc_title, pages in toc_groups.items():
                if len(pages) == 1:
                    sources.append(f"{toc_title} ({pages[0]}페이지)")
                else:
                    # 연속된 페이지 범위로 병합
                    ranges = []
                    start = pages[0]
                    end = pages[0]

                    for i in range(1, len(pages)):
                        if pages[i] == end + 1:
                            end = pages[i]
                        else:
                            if start == end:
                                ranges.append(f"{start}")
                            else:
                                ranges.append(f"{start}-{end}")
                            start = pages[i]
                            end = pages[i]

                    # 마지막 범위 추가
                    if start == end:
                        ranges.append(f"{start}")
                    else:
                        ranges.append(f"{start}-{end}")

                    sources.append(f"{toc_title} ({', '.join(ranges)}페이지)")

            if sources:
                return "\n\n**출처:**\n" + "\n".join(f"- {s}" for s in sources)

            return ""

        except Exception as e:
            self.logger.error(f"Failed to generate source citation: {e}")
            return ""

    async def _get_conversations_by_session_with_context(self, userid: str, sessionid: str) -> List[Dict]:
        """
        Get conversation history from cache

        Returns formatted history for LLM context:
        - recentConversations: Last 10 Q&A pairs with full content
        - previousQuestions: Earlier questions (without answers to save tokens)
        """
        # Get history from conversation cache
        history = await self.conversations.get(sessionid)

        if not history or len(history) == 0:
            return []

        # Format for LLM: recent 10 pairs + older questions only
        recent_conversations = []
        previous_questions = []

        # Group into Q&A pairs
        for i in range(0, len(history), 2):
            if i + 1 < len(history):
                pair = {
                    'question': history[i]['content'],
                    'answer': history[i+1]['content']
                }

                if len(recent_conversations) < 10:
                    recent_conversations.append(pair)
                else:
                    previous_questions.append(history[i]['content'])

        return {
            'recentConversations': recent_conversations,
            'previousQuestions': previous_questions
        }

    async def _get_latest_conversation_context(self, userid: str, sessionid: str) -> Optional[Dict]:
        """
        Get previous search context for followup questions

        Returns METADATA ONLY (no texts to prevent prompt bloat):
        - searchType: Type of search performed
        - pages: Page range used
        - coreKeywords/subKeywords: Keywords used
        - pagesFound: Pages that were found
        - query: Last user question

        NOTE: texts/chunks are NOT stored - will re-fetch if needed
        """
        context_key = f"{userid}:{sessionid}"
        return self.context_store.get(context_key)

    async def _get_toc_core_summary_text(self, pdf_id: str) -> str:
        """Get TOC with core summary (replaces getTocCoreSummaryText)"""
        return await pdf_database.get_toc_core_summary_text(pdf_id)

    async def _get_toc_simple(self, pdf_id: str) -> str:
        """Get simple TOC (replaces getTocSimple)"""
        return await pdf_database.get_toc_simple(pdf_id)

    async def _get_toc_chapters_only(self, pdf_id: str) -> str:
        """Get TOC with chapters only (no subsections)"""
        return await pdf_database.get_toc_chapters_only(pdf_id)

    async def _get_toc_summaries_by_page_spec(self, pdf_id: str, page_spec: str) -> str:
        """Get TOC summaries for page range (replaces getTocSummariesByPageSpec)"""
        return await pdf_database.get_toc_summaries_by_page_spec(pdf_id, page_spec)

    async def _get_chunks_text_by_page_spec(self, pdf_id: str, page_spec: str) -> List[Dict]:
        """Get text chunks by page spec (replaces getChunksTextByPageSpec)"""
        pages = self._parse_page_spec(page_spec)
        if not pages:
            return []

        # 단일 쿼리로 페이지 범위 조회 (N+1 쿼리 방지)
        min_page = min(pages)
        max_page = max(pages)
        all_pages = await pdf_database.get_pages_range(pdf_id, min_page, max_page)

        # 요청된 페이지만 필터링
        page_set = set(pages)
        chunks = [
            {'page': p['page_number'], 'content': p['content']}
            for p in all_pages
            if p['page_number'] in page_set and p['content']
        ]

        return chunks

    def _parse_page_spec(self, page_spec: str) -> List[int]:
        """
        Parse page specification like "1,2,10-20,55"
        Returns list of page numbers
        """
        if not page_spec:
            return []

        pages = []
        parts = page_spec.split(',')

        for part in parts:
            part = part.strip()
            if '-' in part:
                # Range like "10-20"
                start, end = part.split('-')
                try:
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

    async def _fetch_results_with_keywords(
        self,
        pdf_id: str,
        core_keywords: List[Dict],
        sub_keywords: List[Dict],
        pages: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Fetch results using FTS search with keywords
        Replaces pdfService.fetchResults()

        Enhanced with error handling for comparison queries
        """
        try:
            # Build FTS query
            # Core keywords: AND (must all be present)
            # Sub keywords: OR (boost score if present)

            core_terms = []
            for kw in core_keywords:
                keyword = kw['keyword']
                alternatives = kw.get('alternatives', [])
                # Add keyword and all alternatives
                all_terms = [keyword] + alternatives
                core_terms.append(all_terms)

            sub_terms = []
            for kw in sub_keywords:
                keyword = kw['keyword']
                alternatives = kw.get('alternatives', [])
                all_terms = [keyword] + alternatives
                sub_terms.extend(all_terms)

            # Perform FTS search with error handling
            results = await pdf_database.search_pdf_content(
                pdf_id=pdf_id,
                core_keywords=core_terms,
                sub_keywords=sub_terms,
                page_numbers=pages
            )

            chunks = []
            for r in results:
                chunks.append({
                    'page': r['page_number'],
                    'content': r['content']
                })

            self.logger.info(f"Keyword search found {len(chunks)} chunks")

            return {
                'chunks': chunks,
                'visualElements': [],
                'searchLogs': [f"Found {len(chunks)} chunks"]
            }

        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}", exc_info=True)
            # Return empty results instead of crashing
            return {
                'chunks': [],
                'visualElements': [],
                'searchLogs': [f"Search failed: {str(e)}"]
            }

    async def _get_conversation_summary(self, sessionid: str) -> Dict[str, any]:
        """
        세션의 대화 요약 및 최근 대화 가져오기

        Returns:
            {
                "summary": str,  # 전체 대화 요약 (3-5문장)
                "recent_turns": List[Dict]  # 최근 N턴 [{user: ..., assistant: ...}]
            }
        """
        if sessionid not in self.conversation_summaries:
            return {
                "summary": "",
                "recent_turns": []
            }
        return self.conversation_summaries[sessionid]

    async def _format_conversation_context(self, sessionid: str) -> str:
        """
        대화 요약 + 최근 대화를 LLM 프롬프트용 텍스트로 포맷팅

        Returns:
            포맷팅된 대화 컨텍스트 문자열 (대화 요약 + 최근 대화)
        """
        conv_data = await self._get_conversation_summary(sessionid)
        summary = conv_data["summary"]
        recent_turns = conv_data["recent_turns"]

        # 빈 경우 빈 문자열 반환
        if not summary and not recent_turns:
            return ""

        # 포맷팅
        result = ""

        # 1) 대화 요약 블록
        if summary:
            result += f"[대화 요약]\n{summary}\n\n"

        # 2) 최근 대화 블록 (최근 3턴만 표시)
        if recent_turns:
            result += "[최근 대화]\n"
            display_turns = recent_turns[-3:]  # 최근 3턴만 표시
            for i, turn in enumerate(display_turns, 1):
                user_q = turn["user"][:100]  # 너무 길면 축약
                asst_a = turn["assistant"][:150]  # 너무 길면 축약
                result += f"{i}. 사용자: {user_q}\n   답변: {asst_a}\n"

        return result.strip()

    async def _update_conversation_summary(
        self,
        sessionid: str,
        user_query: str,
        assistant_response: str,
        isbn: str
    ):
        """
        대화 요약 및 최근 대화 업데이트

        1. recent_turns에 새 턴 추가 (최대 MAX_RECENT_TURNS개 유지)
        2. LLM으로 전체 대화 요약 생성/갱신
        """
        # 기존 요약 가져오기
        current_data = await self._get_conversation_summary(sessionid)
        current_summary = current_data["summary"]
        recent_turns = current_data["recent_turns"]

        # 1. recent_turns 업데이트 (deque 동작)
        new_turn = {
            "user": user_query,
            "assistant": assistant_response
        }
        recent_turns.append(new_turn)

        # 최대 개수 유지
        if len(recent_turns) > self.MAX_RECENT_TURNS:
            recent_turns = recent_turns[-self.MAX_RECENT_TURNS:]

        # 2. 대화 요약 갱신 (LLM 사용)
        try:
            new_summary = await self._generate_conversation_summary(
                current_summary=current_summary,
                recent_turns=recent_turns,
                isbn=isbn
            )
        except Exception as e:
            self.logger.warning(f"⚠️ 대화 요약 생성 실패, 기존 요약 유지: {e}")
            new_summary = current_summary

        # 3. 저장
        self.conversation_summaries[sessionid] = {
            "summary": new_summary,
            "recent_turns": recent_turns
        }

        self.logger.info(f"📝 Updated conversation summary for {sessionid} (summary: {len(new_summary)} chars, turns: {len(recent_turns)})")

    async def _generate_conversation_summary(
        self,
        current_summary: str,
        recent_turns: List[Dict],
        isbn: str
    ) -> str:
        """
        LLM을 사용하여 대화 전체를 3-5문장으로 요약

        Args:
            current_summary: 현재 요약 (없으면 빈 문자열)
            recent_turns: 최근 대화 턴들
            isbn: 도서 ISBN (컨텍스트용)

        Returns:
            새로운 요약 문자열
        """
        # 최근 대화를 텍스트로 변환
        recent_text = ""
        for i, turn in enumerate(recent_turns[-3:], 1):  # 최근 3턴만 요약에 사용
            recent_text += f"{i}. 사용자: {turn['user']}\n"
            recent_text += f"   답변: {turn['assistant'][:200]}...\n\n"  # 답변은 200자만

        # 요약 생성 프롬프트
        prompt = f"""다음 대화를 3-5문장으로 요약해주세요.

기존 요약:
{current_summary if current_summary else '(없음)'}

최근 대화:
{recent_text}

요구사항:
- 기존 요약이 있으면 이를 참고하여 업데이트
- 사용자가 주로 물어본 주제와 핵심 내용만 간결하게
- 3-5문장으로 작성
- 불필요한 세부사항 제외

요약:"""

        try:
            result = await llm_service.generate_text(
                prompt=prompt,
                mode="chat"
            )
            summary = result['result'].strip()

            # 길이 제한 (최대 500자)
            if len(summary) > 500:
                summary = summary[:500] + "..."

            return summary

        except Exception as e:
            self.logger.error(f"❌ Summary generation failed: {e}")
            # 실패 시 기존 요약 반환
            return current_summary


# Create singleton instance
qa_service = QAService()
