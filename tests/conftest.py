import pytest

from localdecision.backends.mock import CharTokenizer, MockBackend, overlap_scorer


class MergingTokenizer(CharTokenizer):
    """Merges "\\n\\nQuestion" into one token, so prefix/suffix tokenization is NOT additive."""

    MERGED = 0x10FFFF

    def encode(self, text, add_special_tokens=False):
        out, i, key = [], 0, "\n\nQuestion"
        while i < len(text):
            if text.startswith(key, i):
                out.append(self.MERGED)
                i += len(key)
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def decode(self, ids):
        return "".join("\n\nQuestion" if i == self.MERGED else chr(i) for i in ids)


class NoTreeBackend(MockBackend):
    """A backend without child sessions: the engine must fall back to flat scoring."""

    def open(self, prefix):
        session = super().open(prefix)
        session.extend = _raise
        return session


def _raise(tokens):
    raise NotImplementedError


@pytest.fixture
def mock_backend():
    return MockBackend(scorer=overlap_scorer)
