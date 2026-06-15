import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPFeatureExtractor
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_json_file(filepath):
    if isinstance(filepath, str):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif isinstance(filepath, list):
        data = []
        for path in filepath:
            with open(path, 'r', encoding='utf-8') as f:
                data.extend(json.load(f))
        return data
    return []

class ImageTextDataset(Dataset):
    def __init__(self, data, tokenizer, image_processor, image_root, max_length=40, is_eval=False):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_root = image_root
        self.max_length = max_length
        self.is_eval = is_eval

        # 预处理文本和图像路径
        self.samples = []
        categories = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]
        category_to_num = {category: idx for idx, category in enumerate(categories)}
        for sample in tqdm(data, desc='PreProcessing'):
            caption = sample['caption1'] + sample['caption2']
            img_path = sample['image_path']
            image_description = sample['image_description']  # 新增的 image_description
            label = sample['label']  # 获取类别
            category = sample['category']
            # label = sample['labels']  # 获取类别
            category_num = category_to_num.get(category, -1)
            # 分词
            tokenized_caption = self.tokenizer(caption, padding='max_length', truncation=True,
                                               max_length=self.max_length)
            # 分词处理 image_description
            tokenized_image_description = self.tokenizer(image_description, padding='max_length', truncation=True,
                                                         max_length=self.max_length)
            tokenized_caption['img_path'] = img_path
            tokenized_image_description['img_path'] = img_path
            tokenized_caption['label'] = label
            tokenized_image_description['label'] = label
            tokenized_caption['category'] = category_num
            tokenized_image_description['category'] = category_num
            # 将每个样本添加到列表中
            self.samples.append({
                "caption": tokenized_caption,  # 原始 caption 的信息
                "image_description": tokenized_image_description,  # 新增的 image_description 的信息
                "img_path": img_path,
                "label":label,
                "category": category_num,  # 添加类别名
            })
            # tokenized['label'] = label
            # tokenized['raw_caption'] = caption


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        img_name = sample['img_path']
        full_img_path = os.path.join(self.image_root, os.path.basename(img_name))
        try:
            image = Image.open(full_img_path).convert("RGB")
            pixel_values = self.image_processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        except Exception:
            print(f"Warning: failed to load image: {full_img_path}")
            pixel_values = torch.zeros((3, 224, 224))

        # 转为 tensor
        # item = {key: torch.tensor(val) for key, val in sample.items() if key != 'img_path'}
        caption_input_ids = sample['caption']['input_ids']
        caption_attention_mask = sample['caption']['attention_mask']
        label = sample['label']
        category = sample['category']
        image_description_input_ids = sample['image_description']['input_ids']
        image_description_attention_mask = sample['image_description']['attention_mask']
        item = {
            "caption_input_ids": torch.tensor(caption_input_ids),
            "caption_attention_mask": torch.tensor(caption_attention_mask),
            "image_description_input_ids": torch.tensor(image_description_input_ids),
            "image_description_attention_mask": torch.tensor(image_description_attention_mask),
            "pixel_values": pixel_values,
            "label": torch.tensor(label),
            "category": torch.tensor(category)  # 返回类别名
        }

        # ✅ 添加原始文本和图片名，方便调试和验证
        # item['raw_caption'] = sample['raw_caption']
        # item['img_name'] = img_name
        return item

def collate_fn(batch):
    input_keys = batch[0].keys()
    collated = {}
    for key in input_keys:
        if key == 'pixel_values':
            collated[key] = torch.stack([item[key] for item in batch])
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    return collated

def get_dataloaders(args):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

    train_json = load_json_file(args.train_json)
    val_json = load_json_file(args.val_json)
    test_json = load_json_file(args.test_json)

    image_root = args.image_root

    train_dataset = ImageTextDataset(train_json, tokenizer, image_processor, image_root,
                                     max_length=args.text_max_length)
    val_dataset = ImageTextDataset(val_json, tokenizer, image_processor, image_root,
                                   max_length=args.text_max_length, is_eval=True)
    test_dataset = ImageTextDataset(test_json, tokenizer, image_processor, image_root,
                                    max_length=args.text_max_length, is_eval=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
