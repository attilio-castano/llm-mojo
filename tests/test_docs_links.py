"""Documentation links resolve, and every page is reachable from the README."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', 'build'}
# Reachability covers the reader-facing trees; other READMEs link into them freely.
MAPPED = ('docs', 'studies')
# Pages the documentation map does not reach yet.
UNMAPPED = {'studies/decoder_layer/policies_results.md'}
HISTORY_BANNER = '> **Historical record**, kept as written.'


def pages():
    return sorted(p for p in ROOT.rglob('*.md') if not IGNORED & set(p.relative_to(ROOT).parts))


def slug(heading):
    """GitHub's heading anchor: rendered text, lowercased, punctuation removed, spaces to hyphens."""
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', heading.strip())
    text = text.replace('`', '').replace('*', '').lower()
    return re.sub(r'[^\w\- ]', '', text).replace(' ', '-')


def prose(text):
    """Markdown text outside fenced code blocks and inline code spans."""
    text = re.sub(r'^(```|~~~).*?^\1', '', text, flags=re.S | re.M)
    return re.sub(r'`[^`\n]*`', '', text)


def anchors(path):
    found, seen = set(), {}
    for line in prose(path.read_text()).splitlines():
        match = re.match(r'#{1,6}\s+(.*?)\s*#*\s*$', line)
        if match:
            base = slug(match.group(1))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f'{base}-{count}')
    return found


def links(path):
    """(target path, fragment) for every relative link on a page."""
    for target in re.findall(r'\]\(([^)\s]+)(?:\s+"[^"]*")?\)', prose(path.read_text())):
        if re.match(r'[a-z][a-z0-9+.-]*:', target):
            continue
        location, _, fragment = target.partition('#')
        yield ((path.parent / location).resolve() if location else path), fragment


class DocumentationLinkTests(unittest.TestCase):
    def test_relative_links_and_anchors_resolve(self):
        cache, broken, checked = {}, [], 0
        for page in pages():
            for target, fragment in links(page):
                checked += 1
                if not target.exists():
                    broken.append(f'{page.relative_to(ROOT)}: missing {target}')
                elif fragment and target.suffix == '.md':
                    if target not in cache:
                        cache[target] = anchors(target)
                    if fragment not in cache[target]:
                        broken.append(f'{page.relative_to(ROOT)}: no #{fragment} in {target.relative_to(ROOT)}')
        self.assertGreater(checked, 500)
        self.assertEqual(broken, [])

    def test_every_page_is_reachable_from_the_readme(self):
        reached, queue = set(), [ROOT / 'README.md']
        while queue:
            page = queue.pop()
            if page in reached:
                continue
            reached.add(page)
            queue.extend(target for target, _ in links(page) if target.suffix == '.md' and target.exists())
        mapped = {p for p in pages() if p.relative_to(ROOT).parts[0] in MAPPED}
        unreached = {str(p.relative_to(ROOT)) for p in mapped - reached}
        self.assertEqual(unreached, UNMAPPED)

    def test_history_pages_are_labelled(self):
        records = [p for p in (ROOT / 'docs/history').glob('*.md') if p.name != 'README.md']
        self.assertTrue(records)
        for page in records:
            with self.subTest(page=page.name):
                self.assertIn(HISTORY_BANNER, page.read_text().split('\n\n', 2)[1])

    def test_slugs_follow_github_rules(self):
        self.assertEqual(slug('Run the `chat` command'), 'run-the-chat-command')
        self.assertEqual(slug('Correctness and diagnostic policy'), 'correctness-and-diagnostic-policy')
        self.assertEqual(slug('1. Numbers, commas & a [link](x.md)'), '1-numbers-commas--a-link')
        self.assertEqual(slug('decoder_layer (study)'), 'decoder_layer-study')


if __name__ == '__main__':
    unittest.main()
