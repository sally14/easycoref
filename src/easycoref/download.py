import subprocess
from sys import platform
from urllib import request
import os
import zipfile
import tarfile
from pathlib import Path
import logging
import requests

logging.getLogger().setLevel(logging.INFO)

def download():
    subprocess.check_call("python -m spacy download en".split())

    # Create a hidden directory in HOME
    home_path = Path.home()
    easycoref_path = os.path.join(home_path, '.easycoref')
    if os.path.exists(easycoref_path):
        logging.info('Ressources already cached, at least in part')
    else:
        logging.info('Ressources not downloaded yet, starting downloads...')
        os.makedirs(easycoref_path, mode=511)

    # Download pretrained embeddings.
    dst = os.path.join(easycoref_path, 'glove')
    if os.path.exists(os.path.join(dst, 'glove.840B.300d.txt')):
        logging.info('1/3 : Glove Embeddings already downloaded!')

    else:
        logging.info('1/3 : Starting Glove Embeddings Download...')
        url = 'http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip'
        os.makedirs(dst, mode=511, exist_ok=True)
        local_filename, headers = request.urlretrieve(url)
        with zipfile.ZipFile(local_filename,"r") as zip_ref:
            zip_ref.extractall(dst)
        logging.info('Done downloading!')

    # Download char vocab for e2ecoref
    dst = os.path.join(easycoref_path, 'char_vocab.english.txt') 
    if os.path.exists(dst):
        logging.info('2/3 : Char Vocab already downloaded!')
    else:
        logging.info('2/3 : Starting Char Vocab Download...')
        url = 'https://lil.cs.washington.edu/coref/char_vocab.english.txt'
        request.urlretrieve(url, dst)
        logging.info('Done downloading!')

    dst = os.path.join(easycoref_path, 'e2e_logs')
    if not(os.path.exists(dst)):
        os.makedirs(dst, mode=511)

    if any(Path(dst).iterdir()):
        logging.info('3/3 : e2e-coref logs already downloaded!')
    else:
        logging.info('3/3 : Starting e2e-coref logs Download...')
        file_id = "1fkifqZzdzsOEo0DXMzCFjiNXqsKG_cHi"
        destination = os.path.join(dst, 'e2e-coref.tgz')
        print(destination)
        download_file_from_google_drive(file_id, destination)
        logging.info('Done downloading!')
        if os.path.exists(destination):
            logging.info('Unzipping...')
            with tarfile.open(destination,"r:gz") as tar_ref:
                tar_ref.extractall(dst)
                logging.info('Unzipped!')
                os.remove(destination)
        else:
            pass

        logging.info('All done!')

    # Compile tensorflow custom C ops 
    subprocess.check_call("TF_CFLAGS= ( $(python -c 'import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))') )".split())
    subprocess.check_call("TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))') )".split())


    if platform == "linux" or platform == "linux2":
        subprocess.check_call("g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0".split())
    elif platform == "darwin":
        # OS X
        subprocess.check_call("g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -I -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -undefined dynamic_lookup".split())
    else:
        logging.error('Platform not handled, please use a Unix-based os')


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)