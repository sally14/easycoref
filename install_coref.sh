#!/bin/bash

git clone https://github.com/kentonl/e2e-coref.git
cd e2e-coref
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fkifqZzdzsOEo0DXMzCFjiNXqsKG_cHi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fkifqZzdzsOEo0DXMzCFjiNXqsKG_cHi" -O e2e-coref.tgz && rm -rf /tmp/cookies.txt
tar -xzvf e2e-coref.tgz
bash setup_all.sh