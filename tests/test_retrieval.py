from coverwise.ingest import chunk_document


def test_chunks_overlap_and_carry_title(tmp_path):
    doc = tmp_path / "t.md"
    doc.write_text("---\ntitle: T\nsource: s\n---\n" + " ".join(f"w{i}" for i in range(250)), encoding="utf-8")
    chunks = chunk_document(doc, chunk_words=100, overlap=20)
    assert [c.chunk_id for c in chunks] == ["t#0", "t#1", "t#2"]
    assert all(c.text.startswith("T. ") for c in chunks)
    assert "w80" in chunks[0].text and "w80" in chunks[1].text  # overlap


def test_each_mode_finds_the_right_document(retriever):
    for mode in ("bm25", "vector", "hybrid"):
        hits = retriever.search("Are hearing aids covered?", k=2, mode=mode)
        assert hits[0].chunk_id.startswith("hearing-aids"), mode


def test_semantic_match_without_shared_keywords(retriever):
    # "video visit with my doctor" shares no keyword with "telehealth"
    hits = retriever.search("video visit with my doctor from home", k=1, mode="vector")
    assert hits[0].chunk_id.startswith("telehealth")
    assert 0 < hits[0].similarity <= 1
