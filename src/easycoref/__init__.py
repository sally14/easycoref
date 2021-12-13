import subprocess


def download_all():
    subprocess.check_call("python -m spacy download en".split())
    subprocess.check_call("bash install_coref.sh".split())