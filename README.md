# On Efficient Computation of Trajectory Similarity via Single Image Super-Resolution

------

This repository is the official implementation of On Efficient Computation of Trajectory Similarity via Single Image Super-Resolution, based on the *PyTorch*.

## Requirements

- Ubuntu OS
- PyTorch 1.0+
- Python >= 3.5 (Anaconda3 is recommended)

## Preprocessing

------
1. Download dataset 

   ```shell
   curl https://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip -o data/train.csv.zip
   cd data && unzip train.csv.zip
   ```

2. Generate trajectory images
	
	```shell
	python3.7 preprocessing.py
	python3.7 traj2img.py
	```
	
	The hyper parameters used in this experiment are stored in `hyper_parameters.json`, like the selected city area, image sizes and scale of the cell. 
	
	Note that the generated image data may be too large (`200,000` trajectories create image data of nearly 1.3 TB ) to be stored in SSD, and we recommend you to create a soft link to a hard disk drive (HDD), like `ln -s PATH_of_your_HDD/image`, and store these data in HDD. Or you could set a smaller data `amount` in `traj2img.py`(we originally set `amount = 200000`), like `100000`, and accordingly, you also have to change to data size used for training and validation in `dataset.py`.
	
	The generated images are saved in `image/`.

## Training

-----
```shell
python3.7 train.py --cuda --num_epochs 4 --lr 0.0001
```

This procedure will save the models in directory `checkpoint/` at every epoch. We empirically found that 3~4 epochs will produce the model with good enough performance,  and longer training process did not bring much improvement during the inference stage.

You can also use the parameter`--pretrained PATH_TO_PRETRAINED_MODEL`to load trained model. We provide our model `checkpoint_MyG_3_epoch_4.pt`, which is trained after 4 epochs.

## Evaluation

------
Generate embedding vectors for trajectory similarity computation:
```shell
python3.7 generate_data.py --model_path checkpoint/checkpoint_MyG_3_epoch_4.pt
```

Compute the trajectory similarity:

```shell
python3.7 mutilprocess_test.py --model_path checkpoint/checkpoint_MyG_3_epoch_4.pt
```


