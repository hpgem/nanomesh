from pathlib import Path

import nbformat
from jinja2 import Template
from nbconvert import HTMLExporter, RSTExporter

SUBDIR = Path('examples')
OUTDIR = Path(__file__).parent / '_static' / SUBDIR
OUTDIR.mkdir(exist_ok=True, parents=True)

EXPORTERS = {
    '.rst': RSTExporter(),
    '.html': HTMLExporter(),
}

NOTEBOOKS = ('../notebooks/sample_data/multi-domain-plane.ipynb',
             '../notebooks/sample_data/process_raw_data.ipynb',
             '../notebooks/sample_data/sample_data.ipynb',
             '../notebooks/sample_data/sample_data_2D.ipynb',
             '../notebooks/sample_data/sample_data_3D.ipynb',
             '../notebooks/sample_data/select_roi_2d.ipynb')

EXAMPLES_T = Template("""Examples
========

.. toctree::
   :hidden:


{% for object in objects %}

{{ object.body_text }}

See: 🔗 `{{ object.path.name }} <{{ object.link }}>`_

{% endfor %}
""")

REFERENCE_T = Template("""Reference
=========

.. toctree::

{% for object in objects %}
   examples.{{ object.path.stem }}
{%- endfor %}
""")

NOTEBOOK_T = Template("""{{ object.title }}

.. raw:: html
   :file: {{ object.path }}
""")


class NotebookObject:
    def __init__(self, path, notebook):
        self.path = path
        self.notebook = notebook

    @property
    def body_text(self):
        """Grabs the first markdown cell and renders output as `.rst`."""
        for cell in self.notebook.cells:
            if cell['cell_type'] == 'markdown':
                break

        return self.cell_to_rst(cell)

    @property
    def title(self):
        title = self.name
        underline = '=' * len(title)
        return f'{title}\n{underline}'

    @property
    def name(self):
        return self.path.stem

    @property
    def link(self):
        return f'examples.{self.path.name}'

    def cell_to_rst(self, cell):
        """Convert single cell to rst."""
        new = nbformat.v4.new_notebook(
            cells=[cell],
            metadata=self.notebook['metadata'],
            nbformat=self.notebook['nbformat'],
            nbformat_minor=self.notebook['nbformat_minor'])
        exporter = EXPORTERS['.rst']
        (body, resources) = exporter.from_notebook_node(new)

        return body


def notebook_as(path, ext):
    outfile = OUTDIR / path.with_suffix(ext).name
    print('Exporting', outfile)

    notebook = nbformat.read(path, as_version=4)

    exporter = EXPORTERS[ext]
    (body, resources) = exporter.from_notebook_node(notebook)

    with open(outfile, 'w') as f:
        f.write(body)

    return NotebookObject(outfile, notebook)


def notebook2html(path):
    return notebook_as(path, '.html')


def notebook2rst(path):
    return notebook_as(path, '.rst')


def main():
    filenames = (Path(filename) for filename in NOTEBOOKS)

    objects = []

    for filename in filenames:
        # notebook2rst(filename)
        nbobj = notebook2html(filename)

        objects.append(nbobj)
        loc_rst = f'examples.{nbobj.path.stem}.rst'

        with open(loc_rst, 'w') as f:
            s = NOTEBOOK_T.render(object=nbobj)
            f.write(s)

        print('Writing', loc_rst, 'embedding', nbobj.path)

    with open('examples.rst', 'w') as f:
        examples = EXAMPLES_T.render(objects=objects)
        f.writelines(examples)

    with open('examples.reference.rst', 'w') as f:
        reference = REFERENCE_T.render(objects=objects)
        f.writelines(reference)


if __name__ == '__main__':
    main()
