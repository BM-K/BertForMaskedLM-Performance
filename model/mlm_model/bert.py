import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import BertModel
from transformers import BertConfig
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainedModel


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertLMHeadModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        f_outputs=None,
        finetuned=False,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not finetuned:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]
        else:
            sequence_output = f_outputs

        prediction_scores = self.cls(sequence_output)

        preidiction_scores = prediction_scores.view(-1, self.config.vocab_size)

        yhat = preidiction_scores.max(dim=-1)[1]
        y = labels.view(-1)

        correct = (yhat == y).sum()
        total_len = int(y.size(0)) - (y == -100).sum()

        acc = correct / total_len

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(preidiction_scores, y)

        if not return_dict:
            return masked_lm_loss, acc


class BertCLF(nn.Module):
    def __init__(self, args, bert):
        super(BertCLF, self).__init__()

        self.args = args
        self.bert = bert

    def forward(self, inputs, mlm_inputs):

        if self.args.MLM == 'True':
            hidden, pooling = self.bert(input_ids=mlm_inputs['masked_input'],
                                        attention_mask=inputs['attention_mask'],
                                        token_type_ids=inputs['token_type_ids'],
                                        return_dict=False)
        else:
            hidden, pooling = self.bert(input_ids=inputs['input_ids'],
                                        attention_mask=inputs['attention_mask'],
                                        token_type_ids=inputs['token_type_ids'],
                                        return_dict=False)

        return pooling, hidden


class BertMLM(nn.Module):
    def __init__(self, args, bert):
        super(BertMLM, self).__init__()

        self.args = args
        self.bert = bert

    def forward(self, inputs, f_outputs=None, finetuned=False):

        loss, acc = self.bert(input_ids=inputs['masked_input'],
                              labels=inputs['masked_label'],
                              f_outputs=f_outputs,
                              finetuned=finetuned,
                              return_dict=False)

        return loss.data, acc


class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.args = args

        clf_bert = BertModel.from_pretrained('bert-base-uncased')
        mlm_bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

        self.bert_config = BertConfig()
        self.bert_clf = BertCLF(args, clf_bert)
        self.bert_mlm = BertMLM(args, mlm_bert)

        self.projection = nn.Linear(self.bert_config.hidden_size, 2)

    def forward(self, inputs, mlm_inputs):

        if self.args.MLM == 'True':

            with torch.no_grad():

                if self.args.Finetune == 'True':
                    _, hidden = self.bert_clf(inputs, mlm_inputs)
                    loss, acc = self.bert_mlm(mlm_inputs, f_outputs=hidden, finetuned=True)
                else:
                    loss, acc = self.bert_mlm(mlm_inputs)

                return loss, acc

        else:
            poolig_logits, _ = self.bert_clf(inputs)
            logits = self.projection(poolig_logits)

            return logits
