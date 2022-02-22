import json
from pathlib import Path

import pytest

example_dir = Path(__file__).parents[1] / 'docs' / 'examples'
nblink_files = example_dir.glob('*.nblink')


@pytest.mark.parametrize('file', nblink_files)
def test_check_if_nblink_complete(file):
    """Test if nblink files point to an existing notebook."""
    with open(file, 'r') as f:
        data = json.load(f)

    path = Path(data['path'])
    notebook = (file.parent / path)

    assert notebook.exists(
    ), f'{file.name!r} references missing {notebook.name!r}'
