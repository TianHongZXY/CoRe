import os
import jsonlines
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from bert_modeling_base import BertBaseModel
from calculator import batch_calculator_sample as sample


class BertModelForVerifier(BertBaseModel):
    """
    initiates a PyTorch Lightning Bert-like base model for training Verifier, defines training and evaluation steps
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add Bert specific args
        Returns:
            parent_parser
        """
        parser = parent_parser.add_argument_group('BertModelForVerifier')
        parser.add_argument('--verifier_head', default=None, type=str, help="load a saved verifier head model")
        parser.add_argument('--verifier_loss', default="MSE", help="acceptable loss: [MSE, BCE]")

        return parent_parser

    def __init__(self, args, model=None, tokenizer=None, verifier_head=None):
        super().__init__(args, model, tokenizer)
        self.verifier_head = verifier_head
        self.verifier_idx = self.tokenizer.convert_tokens_to_ids("[VERIFIER]")
        if self.hparams.verifier_loss == "BCE":
            assert self.model.num_labels == 1

    def get_inputs(self, batch):
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'token_type_ids': batch['token_type_ids'],
        }
        if self.hparams.verifier_loss == "BCE" and self.hparams.model_type == "deberta":
            inputs['final_token_idx'] = batch['final_token_idx']
        if 'verifier_labels' in batch:
            inputs['verifier_labels'] = batch['verifier_labels']

        return inputs

    def forward(self, input_ids, attention_mask, token_type_ids, verifier_labels=None, final_token_idx=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        if self.hparams.verifier_loss == "MSE":
            verifier_logits = output.logits[:, 0, self.verifier_idx]  # Expected shape = (bs, )
            verifier_predictions = self.verifier_head(verifier_logits.unsqueeze(-1)).squeeze(-1)  # Expected shape = (bs, )

            if verifier_labels is not None:
                loss_fct = nn.MSELoss()
                verifier_loss = loss_fct(verifier_predictions.view(-1), verifier_labels.view(-1))
        elif self.hparams.verifier_loss == "BCE":
            if self.hparams.model_type == "deberta":
                verifier_logits = output.logits.squeeze(-1)
                verifier_logits = torch.gather(verifier_logits, 1, final_token_idx)  # Expected shape = (bs, num_labels)
            else:
                verifier_logits = output.logits[:, 0]  # Expected shape = (bs, num_labels)

            if verifier_labels is not None:
                loss_fct = nn.BCEWithLogitsLoss()
                verifier_loss = loss_fct(verifier_logits.view(-1), verifier_labels.view(-1))
            verifier_predictions = torch.sigmoid(verifier_logits)

        if verifier_labels is not None:
            self.log("verifier_loss", verifier_loss.item(), prog_bar=True, logger=True, on_step=True, batch_size=input_ids.size(0))
            loss = verifier_loss
        else:
            loss = None

        return loss, verifier_predictions

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        batch_size = input_ids.size(0)
        inputs = self.get_inputs(batch)
        del inputs['verifier_labels']
        _, verifier_predictions = self(**inputs)

        verifier_file = os.path.join(self.hparams.data_dir, self.hparams.predict_data) + "_verifier_scored_" + str(self.global_rank)

        with jsonlines.open(verifier_file, 'a') as f:
            for idx in range(batch_size):
                f.write({"question": batch['question'][idx], "solution": batch['solution'][idx] ,"verifier_score": str(verifier_predictions[idx].item()),
                    "is_correct": batch['is_correct'][idx], "question_id": batch['question_id'][idx], "ground_truth": batch['ground_truth'][idx]})

    def save_hf_checkpoint(self) -> None:
        #  if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
        """Save huggingface model checkpoint and tokenizer"""
        if self.global_rank == 0:
            save_path = os.path.join(
                self.trainer.checkpoint_callback.dirpath if self.trainer else self.hparams.save_dir,
                'hf_pretrained_epoch{}_step{}'.format(self.current_epoch, self.global_step))
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            if self.verifier_head:
                torch.save(self.verifier_head, os.path.join(save_path, "verifier_head.pth"))

