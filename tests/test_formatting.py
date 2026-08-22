"""
Regression test for a real bug found while dogfooding the app: Streamlit's
st.markdown() renders `$...$` as inline LaTeX (KaTeX). Campaign plans are
full of dollar figures like "$2.5M" - two unescaped '$' anywhere in the
text turns everything between them into a garbled, letter-spaced math
expression instead of plain text. escape_markdown_dollars() fixes it.
"""

from formatting import escape_markdown_dollars


def test_escapes_dollar_signs():
    assert escape_markdown_dollars(
        "Drive $2.5M in incremental sales on a $300k budget."
    ) == "Drive \\$2.5M in incremental sales on a \\$300k budget."


def test_is_a_no_op_without_dollar_signs():
    assert escape_markdown_dollars("No currency figures here.") == "No currency figures here."


def test_handles_multiple_and_adjacent_dollar_signs():
    assert escape_markdown_dollars("$5, $10, and $$20 combined") == "\\$5, \\$10, and \\$\\$20 combined"
