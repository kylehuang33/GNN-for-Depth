## 1. Set Up


### a. the enviroment
```pip install -r requirement.txt```

### b. Preparation of datasets please follow [the official BTS repo](https://github.com/cleinc/bts)
```shell
$ cd ~/workspace/bts/utils
### Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
### Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
```
## 2. Prelimieries
### a. generating scene graphs
run preprocess/Storing Scene Graphs.ipynb


### b. generate depth maps and embeddings 
run preprocess/Storing the depth embedding.ipynb




### 3. training
##### our pretrained model is [here](https://drive.google.com/drive/folders/1Ej4j3Gj0mjMx9I6OGUsvqzsFYUVa3nQG?usp=sharing)

run Training.ipynb



### 4. Evaluation

run Evaluation.ipynb

