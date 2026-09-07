#!/usr/bin/env python3
"""CPU-only document translation work queue. No LLM calls; no automatic publication.

prepare source.{txt,md,docx,pdf,json} workdir
jobs workdir [--max-chars 3000] [--context-chars 400]
accept workdir results.jsonl
assemble workdir
selftest

Results are JSONL: {id, source_sha256, text, model, issues: []}.
The output is a DRAFT. Complete block coverage is NOT translation-quality approval.
Optional DOCX input: python-docx. Optional PDF input: Poppler pdftotext on PATH.
JSON input: {blocks: [{kind: paragraph|heading|table, text|rows, level?, location?}]}.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = 'docflow/v1'
MAX_BYTES = 32 * 1024 * 1024  # Pilot limit, not a universal document limit.
VALID_ID = re.compile(r'b\d{6}(?:-r\d+-c\d+)?\Z')


def sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode('utf-8') if isinstance(text, str) else text).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: Any) -> None:
    """Atomic replace; callers decide whether replacement is appropriate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        temporary = Path(f.name)
    temporary.replace(path)


def text_blocks(text: str, page: int | None = None) -> list[dict[str, Any]]:
    blocks = []
    for part in re.split(r'\n\s*\n', text.strip()):
        if not part.strip():
            continue
        match = re.fullmatch(r'(#{1,6})\s+([^\n]+)', part.strip())
        block = {'kind': 'heading' if match else 'paragraph',
                 'text': match[2] if match else part.strip(),
                 'location': {'page': page}}
        if match:
            block['level'] = len(match[1])
        blocks.append(block)
    return blocks


def extract(source: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Limited pilot adapters; limitations are retained, not silently hidden."""
    if source.stat().st_size > MAX_BYTES:
        raise ValueError('Pilot file-size limit exceeded; explicitly revise the limit first.')
    suffix = source.suffix.lower()
    if suffix == '.json':
        payload = read_json(source)
        return payload['blocks'], list(payload.get('extraction_issues', []))
    if suffix in {'.txt', '.md'}:
        text = source.read_text(encoding='utf-8')
        blocks = []
        for page, content in enumerate(text.split('\f'), 1):
            blocks.extend(text_blocks(content, page))
        return blocks, ['Plain-text/Markdown adapter preserves text and ATX headings, not full rich formatting.']
    if suffix == '.pdf':
        # No shell, remote URL, generated command, or OCR fallback.
        run = subprocess.run(['pdftotext', '-enc', 'UTF-8', '-layout', str(source.resolve()), '-'],
                             check=True, capture_output=True, timeout=60)
        text = run.stdout.decode('utf-8', errors='strict')
        pages = text.split('\f')
        if pages and not pages[-1].strip():
            pages.pop()
        issues = ['PDF extraction requires source-page inspection; columns, tables, figures and reading order are not validated.']
        blocks = []
        for number, page in enumerate(pages, 1):
            if not page.strip():
                issues.append(f'Page {number}: no extracted text; inspect image/scan/blank page before completion.')
            blocks.extend(text_blocks(page, number))
        return blocks, issues
    if suffix == '.docx':
        import zipfile
        import xml.etree.ElementTree as ET
        from docx import Document
        from docx.table import Table
        # Prevent oversized ZIP expansion in this pilot. Still run parsers in a sandbox.
        with zipfile.ZipFile(source) as archive:
            if sum(i.file_size for i in archive.infolist()) > 4 * MAX_BYTES:
                raise ValueError('DOCX uncompressed-size limit exceeded.')
            xml = archive.read('word/document.xml')
            root = ET.fromstring(xml)
            names = {node.tag.rsplit('}', 1)[-1] for node in root.iter()}
            issues = ['DOCX adapter covers body paragraphs and simple tables only; output does not reproduce source pagination/styles.']
            unsupported = names & {'drawing', 'pict', 'txbxContent', 'footnoteReference',
                                   'endnoteReference', 'ins', 'del', 'fldChar', 'instrText', 'gridSpan', 'vMerge'}
            if unsupported:
                issues.append('Needs source review for: ' + ', '.join(sorted(unsupported)))
            if any(re.match(r'word/(?:header|footer)\d+\.xml$', n) for n in archive.namelist()):
                issues.append('Headers/footers exist and are not included by this pilot adapter.')
        doc = Document(source)
        blocks = []
        if not hasattr(doc, 'iter_inner_content'):
            raise ValueError('DOCX adapter requires python-docx with Document.iter_inner_content().')
        for index, item in enumerate(doc.iter_inner_content()):
            location = {'body_index': index}
            if isinstance(item, Table):
                if any(cell.tables for row in item.rows for cell in row.cells):
                    issues.append(f'Body item {index}: nested table requires separate extraction.')
                blocks.append({'kind': 'table', 'rows': [[cell.text for cell in row.cells] for row in item.rows], 'location': location})
            elif item.text.strip():
                match = re.fullmatch(r'Heading ([1-6])', item.style.name if item.style else '')
                blocks.append({'kind': 'heading' if match else 'paragraph', 'text': item.text,
                               'level': int(match[1]) if match else 1, 'location': location})
        return blocks, issues
    raise ValueError('Supported pilot formats: TXT, Markdown, DOCX, text PDF, normalized JSON.')


def prepare(source: Path, work: Path) -> dict[str, Any]:
    if work.exists():
        raise ValueError('Use a new work directory; never overwrite a previous source manifest.')
    blocks, issues = extract(source)
    if not blocks:
        raise ValueError('No text blocks extracted; inspect the source instead of calling this complete.')
    normalized, segments = [], []
    for index, block in enumerate(blocks):
        kind = block.get('kind')
        if kind not in {'paragraph', 'heading', 'table'}:
            raise ValueError(f'Unsupported block kind {kind!r}; add an explicit adapter, do not drop it.')
        bid = f'b{index:06d}'
        output = {'id': bid, 'kind': kind, 'location': block.get('location', {})}
        if kind == 'table':
            rows = block.get('rows')
            if not rows or not isinstance(rows, list) or any(not isinstance(row, list) or not row for row in rows):
                raise ValueError('Tables require nonempty row lists.')
            if len({len(row) for row in rows}) != 1:
                raise ValueError('Pilot accepts rectangular simple tables only.')
            output['cells'] = []
            for r, row in enumerate(rows):
                cell_ids = []
                for c, text in enumerate(row):
                    if not isinstance(text, str):
                        raise ValueError('Source table cells must be strings.')
                    sid = f'{bid}-r{r}-c{c}'
                    cell_ids.append(sid)
                    segments.append({'id': sid, 'source': text, 'source_sha256': sha(text),
                                     'block_id': bid, 'location': {**output['location'], 'row': r, 'column': c}})
                output['cells'].append(cell_ids)
        else:
            text = block.get('text')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('Text blocks must be nonempty strings.')
            output['segment_id'] = bid
            output['level'] = max(1, min(6, int(block.get('level', 1))))
            segments.append({'id': bid, 'source': text, 'source_sha256': sha(text),
                             'block_id': bid, 'location': output['location']})
        normalized.append(output)
    manifest = {'schema': SCHEMA, 'source_name': source.name, 'source_sha256': sha(source.read_bytes()),
                'blocks': normalized, 'segments': segments, 'extraction_issues': issues}
    work.mkdir(parents=True)
    write_json(work / 'source.json', manifest)
    write_json(work / 'translations.json', {})
    return manifest


def jobs(work: Path, max_chars: int, context_chars: int, glossary: dict[str, Any]) -> list[dict[str, Any]]:
    if max_chars < 1 or context_chars < 0:
        raise ValueError('Invalid character limits.')
    manifest = read_json(work / 'source.json')
    accepted = read_json(work / 'translations.json')
    units = manifest['segments']
    result = []
    by_block = {b['id']: b for b in manifest['blocks']}
    section = ''
    for index, unit in enumerate(units):
        block = by_block[unit['block_id']]
        if block['kind'] == 'heading':
            section = unit['source']
        table_context = None
        if block['kind'] == 'table':
            by_id = {u['id']: u['source'] for u in units if u['block_id'] == block['id']}
            r = unit['location']['row']
            table_context = {'first_row': [by_id[sid] for sid in block['cells'][0]],
                             'current_row': [by_id[sid] for sid in block['cells'][r]]}

        if unit['id'] in accepted or not unit['source'].strip():
            continue
        if len(unit['source']) > max_chars:
            raise ValueError(f"{unit['id']}: exceeds source cap; explicitly split/record coverage, never truncate.")
        # One source block per starter job. The production wrapper can pack complete
        # blocks/sections under the ACTUAL tokenizer budget and add table headers.
        result.append({'id': unit['id'], 'source_sha256': unit['source_sha256'],
                       'source': unit['source'], 'location': unit['location'],
                       'section_context': section, 'table_context': table_context,
                       'context_before': units[index-1]['source'][-context_chars:] if index and context_chars else '',
                       'context_after': units[index+1]['source'][:context_chars] if index+1 < len(units) and context_chars else '',
                       'glossary': glossary, 'source_is_untrusted_data': True,
                       'budget_note': 'Character cap is not a model-token count; tokenize the full prompt and reserve output headroom.'})
    return result


def accept(work: Path, rows: list[dict[str, Any]]) -> None:
    source = read_json(work / 'source.json')
    expected = {u['id']: u for u in source['segments']}
    saved = read_json(work / 'translations.json')
    batch = set()
    for row in rows:
        sid = row.get('id', '')
        if not isinstance(sid, str) or not VALID_ID.fullmatch(sid) or sid not in expected:
            raise ValueError(f'Unknown segment {sid!r}.')
        if sid in batch:
            raise ValueError('Duplicate ID in this result batch.')
        batch.add(sid)
        if row.get('source_sha256') != expected[sid]['source_sha256']:
            raise ValueError(f'{sid}: stale source binding.')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError(f'{sid}: empty/non-text result.')
        if not isinstance(row.get('model'), str) or not row['model'].strip():
            raise ValueError(f'{sid}: model/checkpoint provenance is required.')
        if not isinstance(row.get('issues', []), list):
            raise ValueError(f'{sid}: issues must be a list.')
        if sid in saved and saved[sid] != row:
            raise ValueError(f'{sid}: existing different result; fork a revision workspace explicitly.')
        saved[sid] = row
    write_json(work / 'translations.json', saved)


def audit(work: Path) -> dict[str, Any]:
    source = read_json(work / 'source.json')
    saved = read_json(work / 'translations.json')
    expected = {u['id']: u for u in source['segments']}
    required = {sid for sid, u in expected.items() if u['source'].strip()}
    missing = sorted(required - saved.keys())
    unknown = sorted(saved.keys() - expected.keys())
    invalid = [sid for sid, row in saved.items() if sid in expected and
               (row.get('source_sha256') != expected[sid]['source_sha256'] or not isinstance(row.get('text'), str)
                or not row['text'].strip() or not row.get('model'))]
    # Review flags, NOT semantic grades. Literal digit changes may be legitimate localization.
    flags = []
    for sid, row in saved.items():
        if sid not in expected or sid in invalid:
            continue
        original, translated = expected[sid]['source'], row['text']
        if original == translated:
            flags.append({'id': sid, 'reason': 'unchanged_text; may be an intentional name/code'})
        if sorted(re.findall(r'\d+', original)) != sorted(re.findall(r'\d+', translated)):
            flags.append({'id': sid, 'reason': 'digit_sequences_changed; review locale/meaning'})
        if row.get('issues'):
            flags.append({'id': sid, 'reason': 'worker_reported_issues', 'issues': row['issues']})
    return {'schema': SCHEMA, 'coverage_complete': not (missing or unknown or invalid),
            'required_segments': len(required), 'present_segments': len(required & saved.keys()),
            'missing': missing, 'unknown': unknown, 'invalid': invalid,
            'review_flags': flags, 'extraction_issues': source['extraction_issues'],
            'semantic_quality': 'NOT_ASSESSED', 'publication_authorized': False}


def assemble(work: Path) -> Path:
    report = audit(work)
    write_json(work / 'audit.json', report)
    if not report['coverage_complete']:
        raise ValueError('Cannot assemble: incomplete or invalid source-block coverage.')
    source = read_json(work / 'source.json')
    saved = read_json(work / 'translations.json')
    def content(sid: str) -> str:
        return html.escape(saved[sid]['text']) if sid in saved else ''
    body = ['<h1>Translation draft</h1>', '<p>Complete extracted-block coverage; linguistic and source-layout review still required.</p>']
    for block in source['blocks']:
        bid = block['id']
        if block['kind'] == 'table':
            rows = ['<tr>' + ''.join('<td>' + content(sid) + '</td>' for sid in row) + '</tr>' for row in block['cells']]
            body.append(f'<table id="{bid}">' + ''.join(rows) + '</table>')
        else:
            tag = f"h{block['level']}" if block['kind'] == 'heading' else 'p'
            body.append(f'<{tag} id="{bid}">{content(block["segment_id"])}</{tag}>')
    output = work / 'translation.draft.html'
    output.write_text('<!doctype html><meta charset="utf-8"><title>Translation draft</title>'
                      '<style>body{max-width:52rem;margin:2rem auto;padding:0 1rem;font:16px/1.6 sans-serif}'
                      'p,td{white-space:pre-wrap}table{border-collapse:collapse}td{border:1px solid;padding:.5rem}</style>'
                      + '\n'.join(body), encoding='utf-8')
    return output


def selftest() -> None:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        source = base / 'input.json'
        write_json(source, {'blocks': [{'kind':'heading','text':'Test'}, {'kind':'paragraph','text':'Value 12.'},
                                       {'kind':'table','rows':[['Name','Value'],['A','12'],['','']]}]})
        work = base / 'work'
        manifest = prepare(source, work)
        assert len(jobs(work, 100, 10, {})) == 6
        assert not audit(work)['coverage_complete']
        rows = [{'id': u['id'], 'source_sha256':u['source_sha256'], 'text':'<script>12</script>',
                 'model':'fixture-NOT-A-MODEL', 'issues':[]} for u in manifest['segments'] if u['source'].strip()]
        for bad in [[{**rows[0], 'source_sha256':'wrong'}], [rows[0], rows[0]], [{**rows[0], 'id':'../../escape'}]]:
            try:
                accept(work, bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid result accepted')
        accept(work, rows)
        accept(work, rows)  # Idempotent retry.
        assert jobs(work, 100, 10, {}) == []
        assert audit(work)['coverage_complete']
        output = assemble(work).read_text(encoding='utf-8')
        assert '&lt;script&gt;' in output and '<script>' not in output
        assert audit(work)['semantic_quality'] == 'NOT_ASSESSED'
    print('docflow selftest: PASS (CPU-only; no translation-quality claim)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('source', type=Path); prep.add_argument('work', type=Path)
    job = sub.add_parser('jobs'); job.add_argument('work', type=Path); job.add_argument('--max-chars', type=int, default=3000)
    job.add_argument('--context-chars', type=int, default=400); job.add_argument('--glossary', type=Path)
    acc = sub.add_parser('accept'); acc.add_argument('work', type=Path); acc.add_argument('results', type=Path)
    for name in ('audit', 'assemble'):
        sub.add_parser(name).add_argument('work', type=Path)
    sub.add_parser('selftest')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            value = prepare(args.source, args.work)
            print(json.dumps({'segments': len(value['segments']), 'extraction_issues': value['extraction_issues']}))
        elif args.command == 'jobs':
            for row in jobs(args.work, args.max_chars, args.context_chars, read_json(args.glossary) if args.glossary else {}):
                print(json.dumps(row, ensure_ascii=False))
        elif args.command == 'accept':
            accept(args.work, [json.loads(line) for line in args.results.read_text(encoding='utf-8').splitlines() if line.strip()])
            print('Stored candidate translations; this is not quality approval.')
        elif args.command == 'audit':
            print(json.dumps(audit(args.work), ensure_ascii=False, indent=2))
        elif args.command == 'assemble':
            print(assemble(args.work))
        else:
            selftest()
    except (ValueError, KeyError, OSError, ImportError, subprocess.SubprocessError) as exc:
        parser.exit(2, f'docflow: {exc}\n')

if __name__ == '__main__':
    main()
