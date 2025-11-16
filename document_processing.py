from pathlib import Path
from typing import List, Dict, Tuple
import re
from llama_index.core import SimpleDirectoryReader


class InsuranceDocumentProcessor:

    INSURANCE_KEYWORDS = [
        'coverage', 'deductible', 'copay', 'coinsurance', 'premium',
        'out-of-pocket', 'benefits', 'exclusions', 'limitations',
        'policyholder', 'provider', 'network', 'formulary',
        'prior authorization', 'pre-existing', 'emergency',
        'hospital', 'claim', 'claim form'
    ]


    def extract_text(self, file_path: str) -> List[Dict]:

        file_path = str(file_path)

        directory = str(Path(file_path).parent)
        file_name = Path(file_path).name

        docs = SimpleDirectoryReader(
            input_dir=directory,
            required_exts=[".pdf", ".docx", ".txt", ".md"],
            recursive=False,
            filename_as_id=True
        ).load_data()

        pages: List[Dict] = []

        for doc in docs:
            if doc.metadata.get("file_name") == file_name:

                if "page_label" in doc.metadata:
                    page_no = int(doc.metadata["page_label"])
                    pages.append({
                        "page_no": page_no,
                        "text": doc.text
                    })

                else:
                    pages.append({
                        "page_no": 1,
                        "text": doc.text
                    })

        return pages


    def validate_insurance_document(self, file_path: str) -> Tuple[bool, str]:

        pages = self.extract_text(file_path)
        combined_text = "\n".join([p["text"] for p in pages])

        if not combined_text.strip() or len(combined_text) < 200:
            return False, "Document seems too short or mostly empty."

        matches = sum(
            1 for kw in self.INSURANCE_KEYWORDS
            if kw.lower() in combined_text.lower()
        )

        if matches < 2:
            return False, "Document does not appear strongly insurance-related."

        return True, "Valid insurance-related document."


    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:

        if not text:
            return []

        text = text.replace("\r\n", "\n")

        chunks = []
        start = 0
        n = len(text)

        while start < n:
            end = start + chunk_size
            chunk = text[start:end].strip()
            chunks.append(chunk)
            start = end - overlap
            if start < 0:
                start = 0

        return chunks
