import subprocess
from sys import platform
from urllib.request import urlretrieve
import os
import zipfile
import tarfile
from pathlib import Path
import logging


def download():
    subprocess.check_call("python -m spacy download en".split())

    # Create a hidden directory in HOME
    home_path = Path.home()
    easycoref_path = os.path.join(home_path, '.easycoref')
    os.makedirs(easycoref_path, mode=511, exist_ok=True)

    # Download pretrained embeddings.
    url = 'http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip'
    dst = os.path.join(easycoref_path, 'glove')  # TODO : change embeddings path in e2ecoref
    os.makedirs(dst, mode=511, exist_ok=True)
    local_filename, headers = urlretrieve(url)
    with zipfile.ZipFile(local_filename,"r") as zip_ref:
        zip_ref.extractall(dst)

    # Download char vocab for e2ecoref
    url = 'https://lil.cs.washington.edu/coref/char_vocab.english.txt'
    dst = os.path.join(easycoref_path, 'char_vocab.english.txt') 
    urlretrieve(url, dst)

    
    url_e2elogs = "https://docs.google.com/uc?export=download&id=1fkifqZzdzsOEo0DXMzCFjiNXqsKG_cHi"
    dst = os.path.join(easycoref_path, 'e2e_logs')  # TODO : change embeddings path in e2ecoref
    os.makedirs(dst, mode=511, exist_ok=True)
    local_filename, headers = urlretrieve(url_e2elogs)
    with tarfile.TarFile(local_filename,"r:gz") as tar_ref:
        tar_ref.extractall(dst)


    # Compile tensorflow custom C ops 
    subprocess.check_call("TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )".split())
    subprocess.check_call("TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )".split())


    if platform == "linux" or platform == "linux2":
        subprocess.check_call("g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0".split())
    elif platform == "darwin":
        # OS X
        subprocess.check_call("g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup".split())
    else:
        logging.error('Platform not handled, please use a Unix-based os')

