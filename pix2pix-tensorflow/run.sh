#!/bin/sh

# download dataset
if [ -f photos.tar.gz ]
then
    echo "The file photos.tar.gz already downloaded."
else
    echo "The file photos.tar.gz does not exist. Download now."
    exec wget https://storage.googleapis.com/fivek_dataset/fivek_dataset_for_ELEN6885/photos.tar.gz
fi

# decompress
if [ -d photos ]
then
    echo "Already decompressed."
else
    echo "Start to decompress."
    exec tar -xzvf photos.tar.gz
fi

if [ -f pix2pix.py ]
then
    # run pix2pix.py 
    exec python pix2pix.py --mode train --output_dir photos_train --max_epochs 200 --input_dir photos/combined/train --which_direction BtoA
    exec python pix2pix.py --mode test --output_dir photos_test --input_dir photos/combined/val --checkpoint photos_train
else
    echo "Not found pix2pix.py"
fi
