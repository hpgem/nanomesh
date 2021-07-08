import sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter, RSTExporter

SUBDIR = Path('examples')
OUTDIR = Path(__file__).parent / '_static' / SUBDIR
OUTDIR.mkdir(exist_ok=True)

exporters = {
    '.rst': RSTExporter(),
    '.html': HTMLExporter(),
}


def notebook_as(path, ext):
    outfile = OUTDIR / path.with_suffix(ext).name
    print('Exporting', outfile)

    notebook = nbformat.read(path, as_version=4)

    exporter = exporters[ext]
    (body, resources) = exporter.from_notebook_node(notebook)

    with open(outfile, 'w') as f:
        f.write(body)

    return outfile


def notebook2html(path):
    return notebook_as(path, '.html')


def notebook2rst(path):
    return notebook_as(path, '.rst')


EXAMPLES_TEMPLATE = """
Examples
========

"""

NOTEBOOK_TEMPLATE = """
{title}
=======

.. raw:: html
   :file: {url}
"""


def main():
    examples_rst = EXAMPLES_TEMPLATE

    filenames = (Path(filename) for filename in sys.argv[1:])

    for filename in filenames:
        # notebook2rst(filename)
        html_path = notebook2html(filename)

        loc_html = SUBDIR / html_path.name

        title = loc_html.stem
        examples_rst += f'{title}\n'
        examples_rst += '-' * len(title) + '\n'
        examples_rst += f'`{loc_html.name} <{loc_html}>`_\n'
        examples_rst += '\n'

        loc_rst = f'examples.{html_path.stem}.rst'

        with open(loc_rst, 'w') as f:
            s = NOTEBOOK_TEMPLATE.format(title=title, url=html_path)
            f.write(s)

        print('Writing', loc_rst, 'embedding', html_path)

    with open('examples.rst', 'w') as f:
        f.write(examples_rst)


if __name__ == '__main__':
    main()
