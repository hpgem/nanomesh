This directory stores the jupyter notebooks.

The source for the notebooks are stored in the *markdown* files.
Any code changes should be visible in the source files.
These are converted to notebooks using [jupytext](https://github.com/mwouts/jupytext).
The notebooks are used for documentation via sphinx.

Each notebook is paired to a markdown file, so any changes to the notebook will be automatically saved in both files.

### Convert from ipynb to markdown

    Get-ChildItem "*.ipynb" | Foreach-Object { jupytext $_ --from ipynb --to md }
