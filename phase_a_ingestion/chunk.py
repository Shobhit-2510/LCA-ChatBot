"""Step 3 — split cleaned text into overlapping chunks.

LangChain RecursiveCharacterTextSplitter with the paper's settings
(CHUNK_SIZE=1000, CHUNK_OVERLAP=200 from config). Tag each chunk with
chapter + page metadata so answers can cite their source.

TODO: implement chunk_pages(pages) -> list[Document]
"""
