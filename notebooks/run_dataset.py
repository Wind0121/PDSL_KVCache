from datasets import load_dataset

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

data = load_dataset('json', data_files=f'/home/zk/LongBenchDataset/qasper.jsonl')


data_all = [data[data_sample] for data_sample in data]
for tmp in data_all:
  print(tmp)