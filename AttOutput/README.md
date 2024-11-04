# Hugging Face Transformer 资料
[官方文档](https://huggingface.co/docs/transformers/v4.46.0/en/quicktour)

# 使用流程
## 加载模型
调用 `AutoTokenizer` 和 `AutoModel` 加载分词器和模型。都需要调用 `from_pretrained()` 来获取已有模型的分词器和模型。

```python
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3")
```

## Tokenize
调用 tokenizer 对 prompt 进行 embedding，如下：
```python
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
```

## 文本生成
可以直接调用 model.generate 对 prompt 进行整体生成。也可以自己编写自回归推理过程，从而进行细粒度的控制。此处见代码逻辑

## 绘制注意力分数
每次调用模型都可以获取此次的注意力分数矩阵，以此可以使用 matplotlib 进行绘图

