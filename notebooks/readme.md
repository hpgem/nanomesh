This directory stores the jupyter notebooks.

The source for the notebooks are stored in the *markdown* files.
Any code changes should be visible in the source files.
They can be converted both ways using [jupytext](https://github.com/mwouts/jupytext).
The notebooks are used for documentation via sphinx.

Each notebook is paired to a markdown file, so any changes to the notebook will be automatically saved in both files.

### Convert from ipynb to markdown

    jupytext .\hello_world.ipynb --from ipynb --to md

    Get-ChildItem "**/*.ipynb" | Foreach-Object { jupytext $_ --from ipynb --to md }


### Formatting notebooks with yapf

    jupytext .\hello_world.md --sync --pipe yapf

    Get-ChildItem "**/*.ipynb" | Foreach-Object { jupytext $_ --sync --pipe yapf }


### Execute notebooks commands

Convert notebooks using `jupytext` using the `--execute` flag.

This will use the md file as source. The output is stored in the notebook with the same name.
Any existing ipynb will be overwritten.

    jupytext .\hello_world.md --execute --to ipynb

    Get-ChildItem "**/*.md" | Foreach-Object { jupytext $_ --execute --to ipynb }


### Force sync

This synchronizes any changes to the markdown or ipynb file with each other (inputs only).

    jupytext .\hello_world.ipynb --execute

    Get-ChildItem "**/*.ipynb" | Foreach-Object { jupytext $_ --sync }


### Notebook header

All notebooks must have these magics in the header.
Plots use the *inline* backend so they can be generated programatically.
The figure size is blown up to `10,6`.

```
%config InlineBackend.rc = {'figure.figsize': (10,6)}
%matplotlib inline
```


### Testing notebooks

Powershell:

    ./test_notebooks.PS1

Bash:

    ./test_notebooks.sh
