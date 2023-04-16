import os.path

from ab.utils import logger
from ab.utils.algorithm import algorithm
from ab import app
import torch
from algorithms.utils import Dataset

# 可以使用这种方式载入配置里的数据，然后执行载入模型之类的操作
model_dir = app.config.MODEL_PATH
logger.info('model path is:', model_dir)

# 会自动暴露为/api/algorithm/add接口
@algorithm()
def add(a: int, b: int) -> int:
    """
    一个简单的加法算法示例
    :param a: 第一个参数
    :param b: 第二个参数
    :return:
    """
    import time
    i = 0
    arr = []
    while i < 100:
        arr.append(i)
        time.sleep(1)
        i += 1

    logger.info("enter algorithm {}, {} ".format(a, b))
    return arr

@algorithm()
def encodes_texts(texts, batch_size=64):
    all_vecs = []
    # 加载模型二郎神模型经过SimCSE优化后效果较好
    # 初始化模型
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_p = os.path.join(model_dir, 'IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
    token_p = os.path.join(model_dir, 'IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
    model = AutoModelForMaskedLM.from_pretrained(model_p)
    tokenizer = AutoTokenizer.from_pretrained(token_p)
    net = torch.nn.DataParallel(model)
    model.to("cuda:0")
    def txt2emb(batch_txt):
        with torch.no_grad():
            inp = tokenizer(batch_txt, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
            inp.to("cuda:0")
            out = net(**inp, output_hidden_states=True)
            res = out.hidden_states[-1][:, 0, :].cpu().numpy()

        return res.tolist()

    dataset = Dataset(texts, max_len=512)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    for batch in dataloader:
        vecs = txt2emb(batch)
        all_vecs.extend(vecs)
    torch.cuda.empty_cache()
    return all_vecs