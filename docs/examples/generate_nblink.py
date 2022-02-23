import json
from pathlib import Path

import nbformat

# Use this script to generate nblink files for all
# notebooks `../../notebooks/**.ipynb


def extract_header(path):
    nodes = nbformat.read(path, as_version=nbformat.current_nbformat)
    for cell in nodes['cells']:
        if cell['cell_type'] == 'markdown':
            header = cell['source'].splitlines()[0].strip('# ')
            return header
    raise ValueError(f'Cannot extract header from notebook: {path}')


prefix = Path('../../notebooks')
current_drc = Path(__file__).parent

for path in current_drc.glob('*.nblink'):
    print('Removing', path)
    path.unlink()

notebooks_drc = current_drc.absolute().parents[1] / 'notebooks'
notebooks = notebooks_drc.glob('[!.]*/[!.]*.ipynb')

rubrics = []

for path in notebooks:
    relative_path = path.relative_to(notebooks_drc)
    d = {'path': str(prefix / relative_path)}

    header = extract_header(path)

    rubric = path.parent.name
    rubrics.append(rubric)

    parts = [rubric, *header.lower().split()]
    filename = '_'.join(parts) + '.nblink'

    print('Writing', filename)
    with open(filename, 'w') as f:
        json.dump(d, f)

for rubric in set(rubrics):
    underline = '=' * len(rubric)
    print(f"""
{rubric.capitalize()}
{underline}

.. toctree::
   :maxdepth: 1
   :glob:

   {rubric}*""")
