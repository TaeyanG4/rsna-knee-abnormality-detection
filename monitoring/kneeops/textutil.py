"""Plain-text rendering of Kaggle discussion HTML for search and reading.

The original HTML is preserved in the raw response file; this text is a lossy,
deterministic view. Nothing here executes or fetches linked content.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.links: list[tuple[str, int]] = []
        self.pre = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'br': self.out.append('\n')
        elif tag == 'li': self.out.append('\n- ')
        elif tag in {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}: self.out.append('\n\n' + '#' * int(tag[1]) + ' ')
        elif tag in {'p', 'blockquote'}:
            # A paragraph directly inside a list item continues that bullet.
            if not (self.out and self.out[-1] == '\n- '): self.out.append('\n\n')
        elif tag in {'div', 'tr', 'table', 'ul', 'ol'}: self.out.append('\n')
        elif tag in {'td', 'th'}: self.out.append(' | ')
        elif tag == 'hr': self.out.append('\n---\n')
        elif tag == 'pre':
            self.pre += 1; self.out.append('\n```\n')
        elif tag == 'code' and not self.pre: self.out.append('`')
        elif tag == 'a': self.links.append((a.get('href') or '', len(self.out)))
        elif tag == 'img':
            alt = (a.get('alt') or 'image').strip(); src = a.get('src') or ''
            self.out.append(f'[{alt}]({src})' if src.startswith(('http://', 'https://')) else f'[{alt}]')

    def handle_endtag(self, tag):
        if tag == 'pre':
            self.pre = max(0, self.pre - 1); self.out.append('\n```\n')
        elif tag == 'code' and not self.pre: self.out.append('`')
        elif tag in {'p', 'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}: self.out.append('\n')
        elif tag == 'a' and self.links:
            href, start = self.links.pop()
            text = ''.join(self.out[start:]).strip()
            if href.startswith(('http://', 'https://')) and href.rstrip('/') != text.rstrip('/'):
                self.out.append(f' ({href})')

    def handle_data(self, data):
        if not self.pre: data = re.sub(r'\s+', ' ', data)
        self.out.append(data)


def html_text(value: str | None) -> str:
    if not value: return ''
    if '<' not in value: return value.strip()
    parser = _Text()
    parser.feed(value); parser.close()
    text = ''.join(parser.out)
    text = '\n'.join(line.rstrip() for line in text.splitlines())
    return re.sub(r'\n{3,}', '\n\n', text).strip()
