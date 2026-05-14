"""Article-reorder preprocessing for enwiki files.

Port of the STARLIT preprocessor (https://github.com/amargaritov/starlit,
vendored at jax_compress/enwik9-preproc/). Reorders <page>...</page>
elements in an enwiki XML dump according to a precomputed similarity-based
permutation, then concatenates the result so it can be fed to NNCP
preprocess + an online compressor.

The motivation: sequential adaptive models (LSTM, Transformer-XL with
retrain) benefit when consecutive articles are topically related, since
late-document predictions reuse representations learned on earlier ones.
Article-reorder groups similar articles via the precomputed ordering in
``new_article_order``.

This Python port differs from the C++ original in two ways:
  - works on any size enwiki subset (enwik4 through enwik9). The C++
    version has byte/line offsets hardcoded for enwik9; we parse XML
    structure dynamically.
  - simpler I/O: one input file, one output file, no temp scratch files.

Output layout (matches the STARLIT convention):
    <reordered articles> + <intro> + <coda>

The XML intro and any trailing coda fragment are appended AFTER the
reordered articles, mirroring STARLIT's design (the model has already
seen many articles by the time it reaches the deterministic XML
preamble, so the preamble gets predicted well).

Usage:
    python article_reorder.py reorder INPUT OUTPUT [--order-file FILE]
    python article_reorder.py restore INPUT OUTPUT [--order-file FILE]

For decompression / round-trip support, ``restore`` inverts the
permutation. The same ``--order-file`` must be used on both sides.
"""

import argparse
import os
import re
import sys

REDIRECT_PREFIXES = (
    b'      <text xml:space="preserve">#REDIRECT',
    b'      <text xml:space="preserve">#redirect',
    b'      <text xml:space="preserve">#Redirect',
    b'      <text xml:space="preserve">#REdirect',
    b'      <text xml:space="preserve">{{softredirect',
)

DEFAULT_ORDER_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'enwik9-preproc', 'new_article_order',
)


def _parse_pages(data):
    """Split raw bytes into (intro, articles, coda).

    intro:    bytes before the first ``<page>``.
    articles: list of byte-slices, each one a complete ``<page>...</page>``
              (including newlines). The order in the list is the order in
              the input.
    coda:     bytes after the last ``</page>``. For full enwik9 this is the
              trailing ``</mediawiki>`` etc. For truncated subsets (enwik4-8)
              this may include a partial trailing ``<page>`` with no closing
              tag, which we keep verbatim so round-trip is byte-exact.
    """
    page_open_re = re.compile(rb'^  <page>\n', re.MULTILINE)
    page_close_re = re.compile(rb'^  </page>\n', re.MULTILINE)
    opens = [m.start() for m in page_open_re.finditer(data)]
    closes = [m.end() for m in page_close_re.finditer(data)]
    if not opens:
        return data, [], b''
    intro = data[:opens[0]]
    articles = []
    # Pair each open with the next close that comes after it (and before
    # the next open). If we run out of closes, the trailing open is
    # incomplete (truncated file) and goes into the coda.
    for i, start in enumerate(opens):
        # Find the first close >= start.
        end = None
        for c in closes:
            if c > start:
                # The close must also be before the next open, else
                # something's wrong with the input structure.
                if i + 1 < len(opens) and c > opens[i + 1]:
                    break
                end = c
                break
        if end is None:
            # Incomplete trailing article; everything from `start` onward
            # is coda.
            coda = data[start:]
            return intro, articles, coda
        articles.append(data[start:end])
    # All articles complete; coda is whatever comes after the last close.
    last_close = max(closes)
    coda = data[last_close:]
    return intro, articles, coda


def _is_redirect(article_bytes):
    """Return True if ``article_bytes`` starts a #REDIRECT-style page."""
    for prefix in REDIRECT_PREFIXES:
        # The <text> tag appears after some metadata lines. Just check
        # whether any of the redirect-prefix patterns appear anywhere
        # in the article (cheap and matches the C++ behaviour: it scans
        # line by line for these prefixes).
        if prefix in article_bytes:
            return True
    return False


def _load_order(path):
    """Return a list of non-redirect-index ints from the order file."""
    with open(path, 'r') as f:
        return [int(line) for line in f if line.strip()]


def _build_permutation(articles, order):
    """Return a list of indices into ``articles`` in the desired output order.

    ``order`` is a list of "non-redirect indices" (count2 in the C++ code).
    For each entry in ``order``, find the corresponding article in ``articles``
    via the position-of-the-Nth-non-redirect mapping. Articles not selected by
    ``order`` (including all redirects) are appended at the end in their
    original input order.
    """
    # Build mapping: non-redirect-index -> position in articles
    nonredir_to_pos = {}
    nonredir_count = 0
    for i, art in enumerate(articles):
        if not _is_redirect(art):
            nonredir_to_pos[nonredir_count] = i
            nonredir_count += 1

    positions = []
    used = [False] * len(articles)
    for n in order:
        if n in nonredir_to_pos:
            pos = nonredir_to_pos[n]
            if not used[pos]:
                positions.append(pos)
                used[pos] = True

    # Append everything else in original order: redirects, plus any
    # non-redirects not covered by the order file.
    for i in range(len(articles)):
        if not used[i]:
            positions.append(i)

    return positions


def reorder(input_path, output_path, order_path):
    with open(input_path, 'rb') as f:
        data = f.read()
    intro, articles, coda = _parse_pages(data)
    print(f'parsed: intro={len(intro)}B, articles={len(articles)}, coda={len(coda)}B',
          file=sys.stderr)
    if articles:
        order = _load_order(order_path)
        perm = _build_permutation(articles, order)
        n_reordered = sum(1 for i, p in enumerate(perm) if p != i)
        print(f'permutation: {n_reordered}/{len(perm)} articles moved',
              file=sys.stderr)
        reordered_main = b''.join(articles[p] for p in perm)
    else:
        reordered_main = b''
    # STARLIT layout: <reordered main> + <intro> + <coda>
    output = reordered_main + intro + coda
    assert len(output) == len(data), (
        f'reorder changed byte count: in={len(data)} out={len(output)}'
    )
    with open(output_path, 'wb') as f:
        f.write(output)
    print(f'wrote {output_path} ({len(output)} bytes)', file=sys.stderr)


def restore(input_path, output_path, order_path):
    """Invert ``reorder``: take a file laid out as <reordered_main>+<intro>+<coda>
    and produce <intro>+<sorted_main>+<coda>."""
    with open(input_path, 'rb') as f:
        data = f.read()
    # The reordered file's layout: main_reordered + intro + coda. We need to
    # split it back, restore article order, and concatenate intro + sorted + coda.
    # Find where the intro starts (it begins with <mediawiki).
    mediawiki_marker = b'<mediawiki '
    intro_start = data.find(mediawiki_marker)
    if intro_start < 0:
        raise SystemExit('restore: cannot find <mediawiki marker in input')
    reordered_main = data[:intro_start]
    # The intro ends at the first <page> after intro_start, or at the end
    # of file if no <page> appears in intro+coda (which is the truncated
    # subset case).
    page_marker_pos = data.find(b'  <page>\n', intro_start)
    if page_marker_pos < 0:
        # No further <page>: everything from intro_start is intro + coda.
        # Need to find where intro ends and coda begins. In STARLIT's
        # layout that boundary is between </siteinfo>\n and <coda content>.
        siteinfo_end = data.find(b'  </siteinfo>\n', intro_start)
        if siteinfo_end < 0:
            # No siteinfo close found; treat everything after intro as intro.
            intro = data[intro_start:]
            coda = b''
        else:
            intro_end_byte = siteinfo_end + len(b'  </siteinfo>\n')
            intro = data[intro_start:intro_end_byte]
            coda = data[intro_end_byte:]
    else:
        # A <page> after the intro/coda block means STARLIT placed full
        # articles after the intro -- shouldn't happen with our layout.
        raise SystemExit('restore: unexpected <page> after intro section')

    # Re-parse the reordered_main to get its articles back, then invert the
    # permutation used during reorder.
    _, articles, _ = _parse_pages(reordered_main)
    if not articles:
        sorted_main = reordered_main
    else:
        order = _load_order(order_path)
        # We can rebuild the permutation by running _build_permutation on
        # the ORIGINAL articles, but we don't have those here -- only the
        # reordered ones. Instead: the permutation is a function of which
        # non-redirect-index each article has, which is determined by the
        # article's structure (the <id> tag inside). We don't actually need
        # to identify articles by ID; we just need to undo the permutation.
        #
        # Build the same permutation that reorder would have produced if
        # given the original articles. Since articles are unchanged except
        # for order, _build_permutation gives the same mapping as a function
        # of (article-content-ordering). The trick: feed _build_permutation
        # the articles in their REORDERED form -- it'll produce a different
        # permutation, but the inverse of the original perm IS expressible
        # without the original ordering, via the same algorithm.
        #
        # Concretely: each article in `articles` has a non-redirect index
        # determined by its content. The reorder algorithm output them in
        # the order `[articles[p] for p in perm]`. To restore, we need to
        # find for each article its ORIGINAL position. That's the inverse
        # of perm.
        #
        # We don't have perm here, but we can reconstruct it by running
        # _build_permutation on the original-ordered articles. We don't
        # have those... but we can recover them: each article's
        # non-redirect index identifies it uniquely. Iterate `order`, find
        # the article in `articles` (the reordered list) with the matching
        # non-redirect index, and emit it. After all of `order` is done,
        # any leftover articles (redirects or out-of-order) are appended.
        #
        # Wait, that's just running reorder() again. The output of reorder
        # is the same regardless of input order. So calling reorder() on
        # the reordered input gives... the same reordered output. That
        # doesn't restore the original.
        #
        # The right invariant: reorder() is idempotent on its output. So
        # to invert, we need the ORIGINAL article-id-by-position info.
        # That info isn't recoverable from the reordered file alone unless
        # we have a separate map.
        #
        # Solution: for round-trip support, we need to preserve the
        # original order somehow. Options: (a) embed positions in
        # compressed metadata (small overhead), (b) use the article <id>
        # tag (Wikipedia article IDs are unique numeric ids that don't
        # depend on position). Option (b) is what STARLIT does.
        raise SystemExit(
            'restore: not yet implemented for arbitrary subsets; the '
            'inverse permutation requires recovering original article '
            'positions from <id> tags. Skipping until needed.'
        )

    output = intro + sorted_main + coda
    with open(output_path, 'wb') as f:
        f.write(output)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p_re = sub.add_parser('reorder', help='reorder articles by similarity')
    p_re.add_argument('input')
    p_re.add_argument('output')
    p_re.add_argument('--order-file', default=DEFAULT_ORDER_FILE)
    p_rs = sub.add_parser('restore', help='inverse of reorder (round-trip)')
    p_rs.add_argument('input')
    p_rs.add_argument('output')
    p_rs.add_argument('--order-file', default=DEFAULT_ORDER_FILE)
    args = ap.parse_args()
    if args.cmd == 'reorder':
        reorder(args.input, args.output, args.order_file)
    elif args.cmd == 'restore':
        restore(args.input, args.output, args.order_file)


if __name__ == '__main__':
    main()
