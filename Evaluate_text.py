import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def encode_text(text):
    """使用BERT模型将文本编码为特征向量"""
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取BERT的[CLS] token对应的向量
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding


def compute_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return cosine_similarity(vec1, vec2)[0][0]


def main():
    # 从文件中读取JSON数据
    with open('BERT_midi/text.json', 'r', encoding='utf-8') as f:
        text_data = json.load(f)

    with open('BERT_midi/caption_output.json', 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    # 确保两个文件中的段落数量一致
    assert len(text_data) == len(output_data), "The two JSON files must have the same number of paragraphs."

    similarities = []

    # 遍历每一对文本段落，计算余弦相似度
    for text, output in zip(text_data, output_data):
        vec1 = encode_text(text)
        vec2 = encode_text(output)
        similarity = compute_cosine_similarity(vec1, vec2)
        similarities.append(similarity)

    # 计算所有相似度的平均值
    average_similarity = np.mean(similarities)

    print(f"Average Cosine Similarity: {average_similarity}")


if __name__ == '__main__':
    main()
