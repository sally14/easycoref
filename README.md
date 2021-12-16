# SourcedStatements

## Installation:

Clone this repo and go into the repo:

```bash
git clone git@github.com:sally14/easycoref.git
cd easycoref
```

Create a conda environment in order to isolate the packages needed to run neuralcoref and e2ecoref (we will need a specific version of python, don't mess up with your "root" python!)

```bash
conda create --name Corefenv python==3.7.6 pip
conda activate Corefenv
```

Then install the package and its dependencies with:
```bash
pip install .
```
This is a local install, in the Corefenv conda environment. 

In order to run neuralcoref and e2ecoref, we need to download the ressources provided by the authors. Just run:

```bash
python -m easycoref download
```

Downloading char_vocab.english.txt
```bash
cd ./e2e-coref
curl -O https://lil.cs.washington.edu/coref/char_vocab.english.txt
cd -
```

Enjoy! 
