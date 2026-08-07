"""Body text: HTML comes off before anything is counted, and empty is not short.

``text`` is Hacker News's own name for the body written under the headline. It arrives as
HTML, so a length measured before stripping counts markup, and a vocabulary built before
stripping learns tag names. Both are silent failures: the model still fits.

Synthetic strings only. No data file, no network.
"""

import numpy as np
import pandas as pd

from hn_upvotes.features.body import build_body_features, strip_html, strip_html_column


def test_tags_come_off_and_entities_come_back():
    body = '<p>Read <a href="https://example.com">this</a> &amp; weep &lt;really&gt;.</p>'
    assert strip_html(body) == "Read this & weep <really>."


def test_a_tag_becomes_a_space_rather_than_nothing():
    # `one<p>two` is two words. Deleting the tag outright would invent the word "onetwo"
    # and put it in the vocabulary.
    assert strip_html("one<p>two") == "one two"


def test_an_escaped_bracket_in_the_prose_survives():
    """Entities are unescaped after the tags go, or `&lt;b&gt;` would read as a tag."""
    assert strip_html("a &lt;b&gt; c") == "a <b> c"


def test_a_missing_body_is_the_empty_string():
    # An absent body is a sentinel in this dump, not NULL, so both spellings arrive here.
    assert strip_html(None) == ""
    assert strip_html("") == ""


def test_length_is_measured_on_the_prose_and_not_the_markup():
    plain = strip_html_column(pd.Series(["<p>abcd</p>", "abcd", None]))
    features = build_body_features(plain)
    # All three bodies strip to 4, 4 and 0 characters, so the first two must agree.
    assert features["body_length"].iloc[0] == features["body_length"].iloc[1]
    assert features["body_length"].iloc[0] == np.log1p(4.0)
    assert features["body_length"].iloc[2] == 0.0


def test_no_body_and_a_short_body_are_told_apart():
    """Over 90% of rows have no body. Without the flag they collide with a two-word one."""
    features = build_body_features(strip_html_column(pd.Series(["", "hi"])))
    assert features["has_body_text"].tolist() == [0.0, 1.0]
