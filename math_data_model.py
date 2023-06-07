import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base_data_model import BaseDataModel
from data_preprocess import DataProcessor
from typing import List, Union, Tuple, Optional, Dict, Callable
from pysnooper import snoop


#  ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)\<\|endoftext\|\>")
ANS_RE_opt = re.compile(r"\[ANS\] (\-?[0-9\.\,]+)\<\/s\>")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if not match:
        match = ANS_RE_opt.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        try:
            match_str = float(match_str)
            match_str = round(match_str, 3)
            match_str = str(match_str)
        except:
            print("matched but not a float", match_str)
            return INVALID_ANS
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataModel(BaseDataModel):
    def __init__(self, args, tokenizer): 
        super().__init__(args, tokenizer)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        for idx, ex in enumerate(examples):
            ex.update(question="[QUES]" + ex["question"] + "\n")
            ex.update(answer="[THOUGHT]" + ex["answer"] + self.tokenizer.eos_token)
            ex.update(answer=ex["answer"].replace("####", "[ANS]"))
            ex.update(question_id=str(ex["question_id"]))
            #  ex.update(question_id=str(idx))

        print(f"{len(examples)} examples")
        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer):
        bs = len(batch)
        batch_data = {}
        max_len = 0
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]
        input_ids = []
        attention_mask = []
        labels = []

        for example in batch:
            qns = tokenizer(example['question'], return_attention_mask=False, add_special_tokens=False, max_length=args.source_max_token_len, truncation=True)
            ans = tokenizer(example['answer'], return_attention_mask=False, add_special_tokens=False, max_length=args.target_max_token_len, truncation=True)
            qn_tokens = qns["input_ids"]
            ans_tokens = ans["input_ids"]
            input_ids.append(torch.LongTensor(qn_tokens + ans_tokens))
            attention_mask.append(torch.ones_like(input_ids[-1]))
            if args.loss_on_prefix:
                label = input_ids[-1].clone()
                labels.append(label)
            else:
                label = [-100] * len(qn_tokens) + ans_tokens
                labels.append(torch.LongTensor(label))

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(**batch_data, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def predict_dataloader(self):
        return DataLoader(
            self.custom_dataset(self.raw_predict_data, tokenizer=self.tokenizer),
            batch_size=self.hparams.predict_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        

if __name__ == '__main__':
    a = "[THOUGHT] There are 16 x 3 = <<16*3=48>>48 eggs per day.\n [THOUGHT] Janet’s ducks lay 48 eggs per day. She eats 48 - 3 = <<48-3=45>>45 per day.\nThere are 45 x 4 = <<45*4=180>>180 muffin ingredients.\nShe bakes 180 - 45 = <<180-45=135>>135 muffins.\nShe sells 135 - 48 = <<135-48=87>>87 eggs per day at the farmers' market.\nJanet makes 87 x 2 = $<<87*2=174>>174 every day at the farmers' market.\n[ANS] 174<|endoftext|>"
    print(extract_answer(a))
