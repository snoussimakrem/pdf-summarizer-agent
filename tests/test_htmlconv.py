from pdfsum.dataset.htmlconv import html_to_text


def test_strips_inline_xbrl_hidden_metadata() -> None:
    html = """
    <html><body>
      <ix:header><ix:hidden>http://fasb.org/us-gaap/2025#LongTermDebt</ix:hidden></ix:header>
      <div style="display:none">secret metadata block</div>
      <p>Visible annual report text.</p>
    </body></html>
    """
    text = html_to_text(html)
    assert "fasb.org" not in text
    assert "secret metadata block" not in text
    assert "Visible annual report text." in text
