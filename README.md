# outlier-experiment
Run the following commands to train on UTK-Face dataset
```
# Train and save checkpoint
python utk_train.py --img_dir UTK/UTKFace

# To classify a dataset
python create_facehq.py --img_dir data/celeba_hq/train

# Note img_dir should contain images not folders that contain Images. 
```
