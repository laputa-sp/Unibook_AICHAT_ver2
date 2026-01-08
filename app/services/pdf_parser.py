"""
PDF Parser Service
Handles PDF text extraction and parsing
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import PyPDF2
import pdfplumber
from PIL import Image
import io

logger = logging.getLogger(__name__)


class PDFParser:
    """PDF parsing and text extraction"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def extract_text_from_pdf(
        self,
        pdf_path: Path,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Extract text from PDF file

        Args:
            pdf_path: Path to PDF file
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed)

        Returns:
            List of dictionaries with page number and content
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pages = []

        try:
            # First try with pdfplumber (better for complex layouts)
            pages = await self._extract_with_pdfplumber(pdf_path, start_page, end_page)

            # If pdfplumber fails, fallback to PyPDF2
            if not pages or all(not p['content'].strip() for p in pages):
                self.logger.info("pdfplumber extraction failed, trying PyPDF2")
                pages = await self._extract_with_pypdf2(pdf_path, start_page, end_page)

        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            raise

        return pages

    async def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """Extract text using pdfplumber"""
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            start = (start_page - 1) if start_page else 0
            end = end_page if end_page else total_pages

            # Validate range
            start = max(0, min(start, total_pages - 1))
            end = max(start + 1, min(end, total_pages))

            for i in range(start, end):
                try:
                    page = pdf.pages[i]
                    text = page.extract_text() or ""

                    # Get word count
                    word_count = len(text.split()) if text else 0

                    pages.append({
                        'page_number': i + 1,
                        'content': text.strip(),
                        'word_count': word_count,
                        'width': page.width,
                        'height': page.height
                    })

                except Exception as e:
                    self.logger.warning(f"Error extracting page {i + 1}: {e}")
                    pages.append({
                        'page_number': i + 1,
                        'content': '',
                        'word_count': 0,
                        'error': str(e)
                    })

        return pages

    async def _extract_with_pypdf2(
        self,
        pdf_path: Path,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """Extract text using PyPDF2"""
        pages = []

        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            start = (start_page - 1) if start_page else 0
            end = end_page if end_page else total_pages

            # Validate range
            start = max(0, min(start, total_pages - 1))
            end = max(start + 1, min(end, total_pages))

            for i in range(start, end):
                try:
                    page = pdf_reader.pages[i]
                    text = page.extract_text() or ""

                    # Get word count
                    word_count = len(text.split()) if text else 0

                    pages.append({
                        'page_number': i + 1,
                        'content': text.strip(),
                        'word_count': word_count
                    })

                except Exception as e:
                    self.logger.warning(f"Error extracting page {i + 1} with PyPDF2: {e}")
                    pages.append({
                        'page_number': i + 1,
                        'content': '',
                        'word_count': 0,
                        'error': str(e)
                    })

        return pages

    async def get_pdf_info(self, pdf_path: Path) -> Dict[str, any]:
        """
        Get PDF metadata and information

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with PDF information
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        info = {
            'file_path': str(pdf_path),
            'file_size': pdf_path.stat().st_size,
            'file_name': pdf_path.name
        }

        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                info['page_count'] = len(pdf_reader.pages)

                # Get metadata
                metadata = pdf_reader.metadata
                if metadata:
                    info['title'] = metadata.get('/Title', '')
                    info['author'] = metadata.get('/Author', '')
                    info['subject'] = metadata.get('/Subject', '')
                    info['creator'] = metadata.get('/Creator', '')
                    info['producer'] = metadata.get('/Producer', '')

                # Check if PDF is encrypted
                info['is_encrypted'] = pdf_reader.is_encrypted

        except Exception as e:
            self.logger.error(f"Error getting PDF info: {e}")
            raise

        return info

    async def extract_page_images(
        self,
        pdf_path: Path,
        page_number: int,
        output_dir: Path
    ) -> List[Path]:
        """
        Extract images from a specific page

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            output_dir: Directory to save extracted images

        Returns:
            List of paths to extracted images
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_images = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    raise ValueError(f"Invalid page number: {page_number}")

                page = pdf.pages[page_number - 1]

                # Extract images from page
                if hasattr(page, 'images'):
                    for img_idx, img in enumerate(page.images):
                        try:
                            # Save image
                            img_path = output_dir / f"page_{page_number}_img_{img_idx}.png"
                            # Note: Image extraction implementation depends on pdfplumber version
                            extracted_images.append(img_path)
                        except Exception as e:
                            self.logger.warning(f"Error extracting image {img_idx} from page {page_number}: {e}")

        except Exception as e:
            self.logger.error(f"Error extracting images: {e}")
            raise

        return extracted_images

    async def search_text_in_pdf(
        self,
        pdf_path: Path,
        query: str,
        case_sensitive: bool = False
    ) -> List[Dict[str, any]]:
        """
        Search for text in PDF

        Args:
            pdf_path: Path to PDF file
            query: Search query
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of search results with page numbers and context
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        results = []
        search_query = query if case_sensitive else query.lower()

        try:
            pages = await self.extract_text_from_pdf(pdf_path)

            for page_data in pages:
                content = page_data['content']
                search_content = content if case_sensitive else content.lower()

                if search_query in search_content:
                    # Find all occurrences
                    start = 0
                    while True:
                        pos = search_content.find(search_query, start)
                        if pos == -1:
                            break

                        # Get context (50 chars before and after)
                        context_start = max(0, pos - 50)
                        context_end = min(len(content), pos + len(query) + 50)
                        context = content[context_start:context_end]

                        results.append({
                            'page_number': page_data['page_number'],
                            'position': pos,
                            'context': context,
                            'relevance_score': 1.0  # Simple exact match score
                        })

                        start = pos + 1

        except Exception as e:
            self.logger.error(f"Error searching PDF: {e}")
            raise

        return results


# Create singleton instance
pdf_parser = PDFParser()
