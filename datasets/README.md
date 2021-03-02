## Data

### Shapenet Dataset (GRNet paper, 16284 points)

- dataset structure: category; model-id; 8 views (00.pcd, 01.pcd ....... 07.pcd)
- 也可以自己 Preprocess the [ShapeNet](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz) dataset from PCN paper (没必要,就用 processed 的吧)

```bash
cd /utils
python lmdb_serializer.py /path/to/shapenet/train.lmdb /path/to/output/shapenet/train
python lmdb_serializer.py /path/to/shapenet/valid.lmdb /path/to/output/shapenet/val
```

### 关于数据集的使用：

- Detail Preserved Point Cloud Completion via Separated Feature Aggregation (ECCV 2020): Shapenet
- MSN (AAAI 2020): Shapenet
- GRNet (ECCV 2020): Shapenet, Completion3D, KITTI
- Cascaded Refinement(CVPR 2020): Shapenet, Completion3D 和他们自己建立的 2048points 的数据集

其他：

- ShapeNet 共有 8 类物体
- Completion3D 数据集的 test set 没有提供 category 信息
- KITTI 数据集只有 car 这一个 category。
