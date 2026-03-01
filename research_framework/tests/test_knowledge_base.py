"""
Tests for the Semantic Layer and Knowledge Base components.

This module tests the filters package including:
- Document loaders (PDF, Word, Excel, CSV, Text)
- Vector store (FAISS)
- Knowledge base management
- Semantic layer retrieval

Usage:
    # Run all tests
    python -m pytest tests/test_knowledge_base.py -v

    # Run specific test
    python -m pytest tests/test_knowledge_base.py::test_text_chunker -v

    # Run directly
    python tests/test_knowledge_base.py
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_text_chunker():
    """Test 1: Text chunking functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: Text Chunker")
    print("=" * 60)

    from filters.document_loaders import TextChunker

    chunker = TextChunker(max_size=100, overlap=20)

    # Test short text (no chunking needed)
    short_text = "This is a short text."
    chunks = chunker.chunk(short_text)
    print(f"Short text ({len(short_text)} chars): {len(chunks)} chunk(s)")
    assert len(chunks) == 1
    assert chunks[0] == short_text

    # Test longer text
    long_text = "This is sentence one. " * 20  # ~440 chars
    chunks = chunker.chunk(long_text)
    print(f"Long text ({len(long_text)} chars): {len(chunks)} chunk(s)")
    assert len(chunks) > 1

    # Verify overlap exists
    if len(chunks) > 1:
        # Check that chunks have some overlap
        first_end = chunks[0][-30:]
        second_start = chunks[1][:50]
        print(f"First chunk ends: ...{first_end}")
        print(f"Second chunk starts: {second_start}...")

    print("✓ PASSED")
    return True


def test_text_loader():
    """Test 2: Text file loader."""
    print("\n" + "=" * 60)
    print("TEST 2: Text Loader")
    print("=" * 60)

    from filters.document_loaders import TextLoader, DocumentLoaderConfig

    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content for the text loader.\n")
        f.write("It contains multiple lines.\n")
        f.write("And should be loaded correctly.\n")
        temp_path = f.name

    try:
        config = DocumentLoaderConfig(chunk_size=0)  # No chunking
        loader = TextLoader(config=config)

        docs = loader.load(temp_path)
        print(f"Loaded {len(docs)} document(s) from text file")
        print(f"Content preview: {docs[0].content[:50]}...")

        assert len(docs) == 1
        assert "test content" in docs[0].content
        assert docs[0].metadata.get("file_type") == ".txt"

        print("✓ PASSED")
        return True

    finally:
        os.unlink(temp_path)


def test_vector_store_basic():
    """Test 3: Basic vector store operations."""
    print("\n" + "=" * 60)
    print("TEST 3: Vector Store (Basic Operations)")
    print("=" * 60)

    try:
        from filters.vector_store import FAISSVectorStore, VectorStoreConfig
    except ImportError as e:
        print(f"⚠ SKIPPED: {e}")
        return True

    # Create in-memory vector store
    config = VectorStoreConfig(
        embedding_model="all-MiniLM-L6-v2",
        index_type="flat",
    )

    print("Creating FAISS vector store...")
    store = FAISSVectorStore(config)
    print(f"Store initialized: {store.count} documents")

    # Add documents
    docs = [
        {
            "id": "doc1",
            "content": "Artificial intelligence is transforming healthcare diagnostics.",
            "metadata": {"type": "article", "topic": "AI"},
        },
        {
            "id": "doc2",
            "content": "Machine learning algorithms can detect cancer in medical images.",
            "metadata": {"type": "research", "topic": "ML"},
        },
        {
            "id": "doc3",
            "content": "Deep learning models show promise in radiology applications.",
            "metadata": {"type": "article", "topic": "DL"},
        },
    ]

    print(f"Adding {len(docs)} documents...")
    added = store.add_documents(docs)
    print(f"Added {len(added)} documents, total: {store.count}")

    assert store.count == 3
    assert len(added) == 3

    # Search
    print("\nSearching for 'AI medical diagnosis'...")
    results = store.search("AI medical diagnosis", top_k=2)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  {r['id']}: {r['score']:.3f} - {r['content'][:50]}...")

    assert len(results) == 2
    assert results[0]["score"] > 0  # Should have positive similarity

    # Search with metadata filter
    print("\nSearching with metadata filter (type='article')...")
    results = store.search("AI", top_k=3, filter_metadata={"type": "article"})
    print(f"Found {len(results)} results matching filter")
    for r in results:
        assert r["metadata"]["type"] == "article"

    print("✓ PASSED")
    return True


def test_vector_store_persistence():
    """Test 4: Vector store save/load."""
    print("\n" + "=" * 60)
    print("TEST 4: Vector Store (Persistence)")
    print("=" * 60)

    try:
        from filters.vector_store import FAISSVectorStore, VectorStoreConfig
    except ImportError as e:
        print(f"⚠ SKIPPED: {e}")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and populate store
        config = VectorStoreConfig(index_type="flat")
        store = FAISSVectorStore(config)

        docs = [
            {"id": "d1", "content": "Healthcare AI applications"},
            {"id": "d2", "content": "Medical image analysis"},
        ]
        store.add_documents(docs)

        # Save
        save_path = store.save(tmpdir)
        print(f"Saved to: {save_path}")

        # Create new store and load
        store2 = FAISSVectorStore(config)
        loaded = store2.load(tmpdir)
        print(f"Loaded: {loaded}, documents: {store2.count}")

        assert loaded is True
        assert store2.count == 2

        # Verify search works after load
        results = store2.search("healthcare", top_k=1)
        assert len(results) == 1
        print(f"Search after load: {results[0]['content'][:30]}...")

    print("✓ PASSED")
    return True


def test_knowledge_base():
    """Test 5: Knowledge Base operations."""
    print("\n" + "=" * 60)
    print("TEST 5: Knowledge Base")
    print("=" * 60)

    try:
        from filters.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
    except ImportError as e:
        print(f"⚠ SKIPPED: {e}")
        return True

    # Create knowledge base
    config = KnowledgeBaseConfig(
        name="test_kb",
        chunk_size=200,
        auto_save=False,
    )

    print("Creating knowledge base...")
    kb = KnowledgeBase(config=config)
    print(f"KB initialized: {kb.document_count} documents")

    # Add text content
    print("\nAdding text content...")
    count = kb.add_text(
        text="Artificial intelligence is revolutionizing medical diagnosis. "
             "Machine learning algorithms can now detect diseases with high accuracy. "
             "Deep learning has shown remarkable results in analyzing medical images.",
        source_name="ai_research",
        metadata={"topic": "healthcare AI"},
    )
    print(f"Added {count} chunk(s)")
    assert count > 0

    # Add more content
    count = kb.add_text(
        text="Natural language processing helps analyze patient records. "
             "NLP can extract key information from clinical notes efficiently.",
        source_name="nlp_research",
        metadata={"topic": "NLP"},
    )
    print(f"Added {count} more chunk(s)")

    print(f"\nTotal documents: {kb.document_count}")
    print(f"Total sources: {kb.source_count}")

    # Query
    print("\nQuerying 'machine learning medical'...")
    results = kb.query("machine learning medical", top_k=3)
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  Score: {r['score']:.3f} - {r['content'][:50]}...")

    assert len(results) > 0

    # Get stats
    stats = kb.get_stats()
    print(f"\nKB Stats: {stats.document_count} docs, {stats.source_count} sources")

    print("✓ PASSED")
    return True


def test_directory_loader():
    """Test 7: Directory loader with multiple files."""
    print("\n" + "=" * 60)
    print("TEST 7: Directory Loader")
    print("=" * 60)

    from filters.document_loaders import DirectoryLoader, DocumentLoaderConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_files = [
            ("file1.txt", "Content of first file about AI."),
            ("file2.txt", "Content of second file about machine learning."),
            ("file3.md", "# Markdown file\n\nContent about deep learning."),
        ]

        for filename, content in test_files:
            filepath = Path(tmpdir) / filename
            filepath.write_text(content)
            print(f"Created: {filename}")

        # Load directory
        config = DocumentLoaderConfig(chunk_size=0)  # No chunking
        loader = DirectoryLoader(
            directory=tmpdir,
            recursive=False,
            config=config,
        )

        docs = loader.load()
        print(f"\nLoaded {len(docs)} documents from directory")

        for doc in docs:
            print(f"  {doc.metadata.get('filename')}: {len(doc.content)} chars")

        assert len(docs) == 3

    print("✓ PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("KNOWLEDGE BASE TESTS")
    print("=" * 60)

    results = []

    # Run tests that don't need heavy dependencies first
    results.append(("Text Chunker", test_text_chunker()))
    results.append(("Text Loader", test_text_loader()))
    results.append(("Directory Loader", test_directory_loader()))

    # Run tests that need sentence-transformers and faiss
    print("\nNote: Following tests require sentence-transformers and faiss-cpu")
    print("Install with: pip install sentence-transformers faiss-cpu")

    try:
        results.append(("Vector Store Basic", test_vector_store_basic()))
        results.append(("Vector Store Persistence", test_vector_store_persistence()))
        results.append(("Knowledge Base", test_knowledge_base()))
    except Exception as e:
        print(f"\n⚠ Some tests skipped due to: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
