#!/bin/bash
mkdir -p data
cd data || exit
wget -O datasets.zip 'https://www.dropbox.com/scl/fi/gm3z8v5wxum16cmb84mhq/datasets.zip?rlkey=yf0cd6w378dgdotgndpudt9ep&st=lyexlmuq&dl=1'
unzip datasets.zip
