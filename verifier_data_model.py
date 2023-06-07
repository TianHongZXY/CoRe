import torch
import torch.nn as nn
from base_data_model import BaseDataModel, BaseDataset
from data_preprocess import DataProcessor
from typing import List, Union, Tuple, Optional, Dict, Callable


class GPT2VerifierDataModel(BaseDataModel):
    def __init__(self, args, tokenizer, custom_dataset=BaseDataset):
        super().__init__(args, tokenizer, custom_dataset)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
        print(f"{len(examples)} examples")

        return examples

    @staticmethod
    def collate_fn(batch, args, tokenizer):
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        qn_sol_input_ids = []
        qn_ans_input_ids = []
        qn_sol_mask = []
        qn_ans_mask = []
        labels = []
        verifier_labels = []
        verifier_loss_mask = []
        final_token_idx = []
        for example in batch:
            qns = tokenizer(example['question'])
            sol = tokenizer(example['solution'])
            ans = tokenizer(example['ground_truth'])

            qn_sol_input_ids.append(torch.LongTensor(qns.input_ids + sol.input_ids))
            qn_ans_input_ids.append(torch.LongTensor(qns.input_ids + ans.input_ids))
            qn_sol_mask.append(torch.ones_like(qn_sol_input_ids[-1]))
            qn_ans_mask.append(torch.ones_like(qn_ans_input_ids[-1]))

            final_token_idx.append(len(qn_sol_input_ids[-1]) - 1)

            if args.lm_objective:
                label = torch.LongTensor([-100] * len(qns.input_ids) + ans.input_ids)
                labels.append(label)
            else:
                labels = None

            verifier_label = torch.ones_like(qn_sol_input_ids[-1]) * float(example['is_correct'])
            verifier_labels.append(verifier_label)
            verifier_mask = [0] * len(qns.input_ids) + [1] * len(sol.input_ids) # + [0] * sol_pad_length
            verifier_loss_mask.append(torch.LongTensor(verifier_mask))

        qn_sol_input_ids = nn.utils.rnn.pad_sequence(qn_sol_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        qn_ans_input_ids = nn.utils.rnn.pad_sequence(qn_ans_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        qn_sol_mask = nn.utils.rnn.pad_sequence(qn_sol_mask, batch_first=True, padding_value=0)
        qn_ans_mask = nn.utils.rnn.pad_sequence(qn_ans_mask, batch_first=True, padding_value=0)
        verifier_labels = nn.utils.rnn.pad_sequence(verifier_labels, batch_first=True, padding_value=-100)
        verifier_loss_mask = nn.utils.rnn.pad_sequence(verifier_loss_mask, batch_first=True, padding_value=0)
        final_token_idx = torch.LongTensor(final_token_idx).view(-1, 1)
        if labels:
            labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(**batch_data,
                    qn_ans_input_ids=qn_ans_input_ids, qn_ans_mask=qn_ans_mask, labels=labels,
                    qn_sol_input_ids=qn_sol_input_ids, qn_sol_mask=qn_sol_mask, verifier_labels=verifier_labels,
                    verifier_loss_mask=verifier_loss_mask,
                    )


class VerifierPredictDataModel(BaseDataModel):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)

    def get_examples(self, path, type):
        examples = DataProcessor._read_jsonl(path)
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
        final_token_idx = []

        for example in batch:
            qns = tokenizer(example['question'], return_attention_mask=False)
            sol = tokenizer(example['solution'], return_attention_mask=False)
            qn_tokens = qns["input_ids"]
            sol_tokens = sol["input_ids"]

            input_ids.append(torch.LongTensor(qn_tokens + sol_tokens))
            attention_mask.append(torch.ones_like(input_ids[-1]))
            final_token_idx.append(len(qn_tokens + sol_tokens) - 1)

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        final_token_idx = torch.LongTensor(final_token_idx).view(-1, 1)

        return dict(**batch_data, input_ids=input_ids, attention_mask=attention_mask, final_token_idx=final_token_idx)


