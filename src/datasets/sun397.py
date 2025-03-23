import os
import torch
import torchvision.datasets as datasets

class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # 数据加载代码
        traindir = os.path.join(location, 'SUN397_splits', 'train')
        testdir = os.path.join(location, 'SUN397_splits', 'test')

        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(testdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # 根据数据集名称索引
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]
        
        # 为验证集分割创建class_splits属性
        self.class_splits = self.load_class_splits(location)
    
    def split_class_data(self, train):
        """找到与每个类对应的数据索引"""
        indices = {}
        dataset = self.train_dataset if train else self.test_dataset
        for i, (_, t) in enumerate(dataset):
            if t not in indices:
                indices[t] = [i,]
            else:
                indices[t].append(i)
        return indices

    def load_class_splits(self, location):
        """加载每个类的数据索引列表"""
        root_dir = os.path.join(location, 'SUN397')
        cache_path = os.path.join(root_dir, 'class_splits.json')
        
        import json
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                class_splits = json.load(f)
            return class_splits
        else:
            print(
                f"SUN397的类分割未找到。"
                "\n生成并缓存类分割..."
            )
            class_splits = {
                'train': self.split_class_data(True),
                'test': self.split_class_data(False),
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            with open(cache_path, 'w') as f:
                json.dump(class_splits, f)
            return class_splits


class SUN397Val(SUN397):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, location, batch_size, num_workers)
        
        # 验证集覆盖测试集
        valdir = os.path.join(location, 'SUN397_splits', 'val')
        if os.path.exists(valdir):
            self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )