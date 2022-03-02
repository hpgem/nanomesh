This directory stores the jupyter notebooks.

The source for the notebooks are stored in the *markdown* files.
Any code changes should be visible in the source files.
They can be converted both ways using [jupytext](https://github.com/mwouts/jupytext).
The notebooks are used for documentation via sphinx.

Each notebook is paired to a markdown file, so any changes to the notebook will be automatically saved in both files.

### Convert from ipynb to markdown

    jupytext .\hello_world.ipynb --from ipynb --to md

    Get-ChildItem "**/*.ipynb" | Foreach-Object { jupytext $_ --from ipynb --to md }


### Formatting with yapf

    jupytext .\hello_world.md --sync --pipe yapf

    Get-ChildItem "**/*.ipynb" | Foreach-Object { jupytext $_ --sync --pipe yapf }


### Execute notebooks commands

    jupytext .\hello_world.ipynb --execute

    Get-ChildItem "**/*.ipynb" | Foreach-Object { jupytext $_ --execute }
