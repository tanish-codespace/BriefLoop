"""
Small formatting helpers with no Streamlit dependency, kept separate from
app.py so they're trivially unit-testable (importing app.py runs the whole
Streamlit script top-to-bottom, which is a bad place to import a helper
function from in a test).
"""


def escape_markdown_dollars(text: str) -> str:
    """Escape bare '$' before handing LLM-generated text to st.markdown().

    Streamlit's markdown renderer treats `$...$` as inline LaTeX (KaTeX).
    Campaign plans are full of dollar figures like "$2.5M" - two unescaped
    '$' anywhere in the text turns everything between them into a garbled
    math expression instead of showing the dollar amounts. Escaping to
    '\\$' keeps them as plain text without affecting any other formatting.
    """
    return text.replace("$", "\\$")
