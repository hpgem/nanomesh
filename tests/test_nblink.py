import json
from pathlib import Path


def test_check_if_nblink_complete():
    """Test if nblink files point to an existing notebook."""
    example_dir = Path(__file__).parents[1] / 'docs' / 'examples'

    nblink_files = example_dir.glob('*.nblink')

    for file in nblink_files:
        with open(file, 'r') as f:
            data = json.load(f)

        path = Path(data['path'])

        assert (file.parent / path).exists()
