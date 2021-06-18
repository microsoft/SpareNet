## Data

### Shapenet Dataset (16284 points)

- Download the processed dataset from [GRNet](https://github.com/hzxie/GRNet): https://gateway.infinitescript.com/?fileName=ShapeNetCompletion 
    - dataset structure: category; model-id; 8 views (00.pcd, 01.pcd ....... 07.pcd)
- Or you can preprocess the [ShapeNet](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz) dataset from [PCN](https://github.com/wentaoyuan/pcn/tree/master/data) 

    ```bash
    cd /utils
    python lmdb_serializer.py /path/to/shapenet/train.lmdb /path/to/output/shapenet/train
    python lmdb_serializer.py /path/to/shapenet/valid.lmdb /path/to/output/shapenet/val
    ```
where `lmdb_serializer.py` can be obtained [here](https://github.com/hzxie/GRNet/blob/master/utils/lmdb_serializer.py).

### KITTI Dataset
- Download the dataset: https://drive.google.com/drive/folders/1fSu0_huWhticAlzLh3Ejpg8zxzqO1z-F