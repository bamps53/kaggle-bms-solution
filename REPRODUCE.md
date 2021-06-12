# 0.Environment
You can use any of these environmet.
### Preprocess
1. Kaggle kernel
2. Kaggle docker image

### Training
1. Kaggle kernel(TPU)
2. Google colab(TPU)

### Inference
1. Kaggle kernel(GPU)
2. Google colab(GPU)

Also you need to configure your google cloud storage bucket to save model weights.
If you get it, replace ```YOUR_GCS_DIR``` in main script with it.

# 1. Preprocess

### 1-1.setup
​
At first, it is required to download and put the BMS competition data into input directory like this:
​
```
bms/input/bms-molecular-translation/
├── extra_approved_InChIs.csv
├── sample_submission.csv
├── test
├── train
└── train_labels.csv
```

### 1-2.preprocess
```
python preprocess/create_folds.py
```

### 1-3.resize images
```
python preprocess/resize_train_images.py -h 300 -w 600
python preprocess/resize_train_images.py -h 416 -w 736
python preprocess/resize_train_images_noise_denoise.py -h 416 -w 736

python preprocess/resize_test_images.py -h 300 -w 600
python preprocess/resize_test_images.py -h 416 -w 736
python preprocess/resize_test_images_noise_denoise.py -h 416 -w 736

```

### 1-4.create tfrecords
```
python preprocess/create_train_tfrecords.py -d data/folds.csv -i data/resized300x600 -s data/tfrecords013
python preprocess/create_train_tfrecords.py -d data/folds.csv -i data/resized416x736 -s data/tfrecords018
python preprocess/create_train_tfrecords.py -d data/lb060.csv -i data/resized416x736 -s data/tfrecords031
python preprocess/create_train_tfrecords.py -d data/folds.csv -i data/resized416x736_noise_denoise -s data/tfrecords032
python preprocess/create_test_tfrecords.py -d ../input/bms-molecular-translation/sample_submission.csv -i data/resized300x600_test -s data/tfrecords016
python preprocess/create_test_tfrecords.py -d ../input/bms-molecular-translation/sample_submission.csv -i data/resized416x736_test -s data/tfrecords020
python preprocess/create_test_tfrecords.py -d ../input/bms-molecular-translation/sample_submission.csv -i data/resized416x736_noise_denoise_test -s data/tfrecords033
```

### 1-5.upload to kaggle datasets
They're already uploaded to public dataset.  
https://www.kaggle.com/bamps53/tfrecords013  
https://www.kaggle.com/bamps53/tfrecords016  
https://www.kaggle.com/bamps53/tfrecords018  
https://www.kaggle.com/bamps53/tfrecords020  
https://www.kaggle.com/bamps53/tfrecords031
https://www.kaggle.com/bamps53/tfrecords031-2
https://www.kaggle.com/bamps53/tfrecords032  
https://www.kaggle.com/bamps53/tfrecords033  

### 1-6. get gcs path in kaggle notebook
It's done in this kernel.
https://www.kaggle.com/bamps53/bms-get-gcs-path
Then paste them in main scripts.

# 2. Training
Keep running these scripts until reaching end of epochs
```
python main.py -c exp/072.yaml -m train
python main.py -c exp/084.yaml -m train
python main.py -c exp/0845.yaml -m train
python main.py -c exp/090.yaml -m train
python main.py -c exp/103.yaml -m train
python main.py -c exp/1031.yaml -m train
```

# 3. Inference
After finishing training, set ```inference = True``` in ```Config``` class in main scripts.
Then rerun all.
```
python 02_main/exp072.py
python 02_main/exp084.py
python 02_main/exp090.py
python 02_main/exp103.py
python 02_main/exp1031.py
python 02_main/exp0845.py
```

# 4. Rescore
After downloading submission files from Kaggle dataset, run these scripts to rescore candidates.
```
python 03_rescore/exp072.py
python 03_rescore/exp084.py
python 03_rescore/exp090.py
python 03_rescore/exp103.py
python 03_rescore/exp1031.py
python 03_rescore/exp0845.py
```