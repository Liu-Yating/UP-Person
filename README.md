# UP-Person: Unified Parameter-Efficient Transfer Learning for Text-based Person Retrieval


## Highlights

The goal of this work is to design a unified parameter-efficient transfer learning method for text-based person retrieval,  which effectively transfers both local and global knowledge, along with task-specific knowledge, to TPR task with very fewer computation and storage costs.
## Usage
### Requirements
we use single NVIDIA 4090 24G GPU for training and evaluation. 
```
pytorch 1.12.1
torchvision 0.13.1
prettytable
easydict
```

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset)

Organize them in `your dataset root dir` folder as follows:
```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```


## Training

``` python train.py \
--name baseline \
--img_aug \
--batch_size 128 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm' \
--num_epoch 60 \
--root_dir '.../dataset_reid' \
--lr 1e-3 \
--prefix_length 10 \
--rank 32 \
--depth_lora 12 \
--depth_prefix 12 \
--depth_adapter 0 
```

## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```


## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [IRRA](https://github.com/anosorae/IRRA) and [LAE](https://github.com/gqk/LAE). We sincerely appreciate for their contributions.

