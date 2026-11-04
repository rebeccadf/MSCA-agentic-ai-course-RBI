import rag


def test_clean_filing_joins_table_columns():
    text = "  Total current liabilities      7,118      5,964  \n\n   \nRevenue   100   90"
    assert rag.clean_filing(text) == "Total current liabilities | 7,118 | 5,964\nRevenue | 100 | 90"


def test_clean_filing_keeps_single_and_double_spaces():
    assert rag.clean_filing("Net  income was up") == "Net  income was up"


def test_tokenize_lowercases_and_drops_punctuation():
    assert rag.tokenize("Total Current-Liabilities: $7,118!") == ["total", "current", "liabilities", "7", "118"]


def test_find_phrase_ignores_case_and_punctuation(make_retriever):
    retriever = make_retriever([
        "Revenue | 100 | 90",
        "TOTAL CURRENT LIABILITIES | 7,118 | 5,964",
    ])
    [node] = retriever.find_phrase("total current liabilities", "ACME_2023_10K")
    assert "7,118" in node.get_content()


def test_find_phrase_matches_whole_words_only(make_retriever):
    retriever = make_retriever(["Total current liabilitiesX | 1"])
    assert retriever.find_phrase("total current liabilities", "ACME_2023_10K") == []


def test_find_phrase_ranks_by_occurrences_and_limits(make_retriever):
    retriever = make_retriever([
        "capex once",
        "capex 2023, capex 2022, capex 2021",
        "capex 2023, capex 2022",
        "nothing here",
    ])
    nodes = retriever.find_phrase("capex", "ACME_2023_10K", max_chunks=2)
    assert [n.get_content() for n in nodes] == ["capex 2023, capex 2022, capex 2021", "capex 2023, capex 2022"]


def test_find_phrase_empty_phrase_returns_nothing(make_retriever):
    retriever = make_retriever(["anything"])
    assert retriever.find_phrase("  --  ", "ACME_2023_10K") == []
