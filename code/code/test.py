import torch
from collections import Counter
from typing import List

class BasicTokenizer:
    def __init__(self, corpus: List[str] = None):
        """
        Tokenizer的构造函数。
        可以传入一个语料库来直接构建词汇表。

        Args:
            corpus (List[str], optional): 一个句子列表，用于构建词汇表。Defaults to None.
        """
        # 1. 初始化特殊符号和词汇表
        self.special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        self.vocab = {}
        self.reverse_vocab = {}
        
        # 立即为特殊符号分配固定的ID
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
            
        if corpus:
            self.build_vocab(corpus)

    def build_vocab(self, corpus: List[str]):
        """
        从语料库构建词汇表。

        Args:
            corpus (List[str]): 句子列表。
        """
        # 下一个可用ID从特殊符号数量之后开始
        next_id = len(self.special_tokens)
        
        # 将所有句子转换成小写并拆分单词
        words = []
        for sentence in corpus:
            words.extend(self._tokenize_text(sentence))
            
        # 统计词频，可以用于之后过滤低频词（本例中未使用）
        word_counts = Counter(words)
        
        # 遍历所有唯一的词，构建词汇表
        for word in word_counts.keys():
            if word not in self.vocab:
                self.vocab[word] = next_id
                self.reverse_vocab[next_id] = word
                next_id += 1
        
        print(f"词汇表构建完成！总词数: {len(self.vocab)}")

    def _tokenize_text(self, text: str) -> List[str]:
        """一个非常基础的分词器，按空格切分并转为小写。"""
        return text.lower().split(' ')

    def encode(self, text: str) -> torch.Tensor:
        """
        将文本字符串编码为PyTorch Tensor。

        Args:
            text (str): 输入的句子。

        Returns:
            torch.Tensor: 编码后的整数ID张量。
        """
        encoded_ids = []
        
        # 1. 在句子开头添加<sos>
        encoded_ids.append(self.vocab['<sos>'])
        
        # 2. 处理句子中的每个词
        words = self._tokenize_text(text)
        for word in words:
            # 使用.get()方法处理未知词(OOV)，如果词不在词汇表中，则返回<unk>的ID
            token_id = self.vocab.get(word, self.vocab['<unk>'])
            encoded_ids.append(token_id)
            
        # 3. 在句子结尾添加<eos>
        encoded_ids.append(self.vocab['<eos>'])
        
        # 4. 使用PyTorch基础语法将列表转换为Tensor
        return torch.tensor(encoded_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor or List[int]) -> str:
        """
        将Tensor或ID列表解码为文本字符串。

        Args:
            token_ids (torch.Tensor or List[int]): token ID序列。

        Returns:
            str: 解码后的句子。
        """
        # 如果输入是Tensor，先转换为Python列表
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        decoded_words = []
        for token_id in token_ids:
            # 查找ID对应的词，如果ID未知则返回<unk>
            word = self.reverse_vocab.get(token_id, '<unk>')
            decoded_words.append(word)
        
        # 过滤掉特殊符号，然后将词语拼接成句子
        filtered_words = [
            word for word in decoded_words 
            if word not in self.special_tokens
        ]
        
        return ' '.join(filtered_words)

# --- 示例用法 ---

# 1. 准备一个简单的语料库
corpus = [
    "hello world",
    "this is a simple tokenizer",
    "hello pytorch a simple world"
]

# 2. 创建并训练Tokenizer
tokenizer = BasicTokenizer(corpus)

# 3. 查看构建好的词汇表
print("\n--- 词汇表示例 ---")
print(tokenizer.vocab)
# 预期输出: {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3, 'hello': 4, 'world': 5, 'this': 6, ...}


# 4. 演示编码(Encode)功能
print("\n--- 编码演示 ---")
text_to_encode = "hello this is a new world"
encoded_tensor = tokenizer.encode(text_to_encode)
print(f"原文: '{text_to_encode}'")
print(f"编码后的Tensor: {encoded_tensor}")
# 预期输出: tensor([ 2,  4,  6,  7,  8,  1,  5,  3])
# 解释: 2(<sos>), 4(hello), 6(this), 7(is), 8(a), 1(<unk>因为'new'是未知词), 5(world), 3(<eos>)


# 5. 演示解码(Decode)功能
print("\n--- 解码演示 ---")
decoded_text = tokenizer.decode(encoded_tensor)
print(f"从Tensor解码后的原文: '{decoded_text}'")
# 预期输出: 'hello this is a <unk> world'


# 6. 演示对纯列表的解码
print("\n--- 对列表解码演示 ---")
id_list = [2, 8, 9, 6, 10, 3]
decoded_from_list = tokenizer.decode(id_list)
print(f"从列表 {id_list} 解码: '{decoded_from_list}'")
# 预期输出: 'a simple is tokenizer'