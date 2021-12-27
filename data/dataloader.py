import re
import torch
import logging
import random
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer):
        self.args = args
        self.metric = metric

        self.label = []
        self.sentence = []
        self.token_type_ids = []
        self.attention_mask = []

        self.masked_input = []
        self.masked_labels = []

        self.file_path = file_path
        self.bert_tokenizer = tokenizer
        self.vocab_size = len(self.bert_tokenizer)
        self.masking_ratio = self.args.masking_ratio

        """
        init token, idx = [CLS], 101
        sep token, idx = [SEP], 102
        pad token, idx = [PAD], 0
        unk token, idx = [UNK], 100
        eos token, idx = [EOS], 30522
        max token, idx = [MASK], 103
        """
        self.init_token = self.bert_tokenizer.cls_token
        self.sep_token = self.bert_tokenizer.sep_token
        self.pad_token = self.bert_tokenizer.pad_token
        self.mask_token = self.bert_tokenizer.mask_token
    
        self.init_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.init_token)
        self.sep_toke_idx = self.bert_tokenizer.convert_tokens_to_ids(self.sep_token)
        self.pad_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.pad_token)
        self.mask_token_idx = self.bert_tokenizer.convert_tokens_to_ids(self.mask_token)

    # Load train, valid, test data in args.path_to_data
    def load_data(self, type):

        with open(self.file_path) as file:
            lines = file.readlines()

            for line in lines:

                sentence, token_type, attention_mask, label = self.data2tensor(line)
                
                if sentence == None:
                    continue

                self.sentence.append(sentence)
                self.token_type_ids.append(token_type)
                self.attention_mask.append(attention_mask)
                self.label.append(label)

        assert len(self.sentence) == \
               len(self.label)

    def preprocess_text(self, review):
        review = re.sub("<br />", " ", review)
        alphanum = re.sub("[^a-zA-Z0-9]", " ", review)
        words = alphanum.lower().split()
        tokens = words
        return (" ".join(tokens))

    """
    Converting text data to tensor &
    expanding length of sentence to args.max_len filled with PAD idx
    """
    def data2tensor(self, line):
        split_data = line.split('\t')

        sentence, label = self.preprocess_text(split_data[0].strip()), split_data[1].strip()

        if len(sentence) == 0:
            return None, 0, 0, 0
        
        sentence_tokens = self.bert_tokenizer(sentence,
                                              return_tensors="pt",
                                              max_length=self.args.max_len,
                                              pad_to_max_length="right")
    
        input_ids = sentence_tokens['input_ids']
        token_type_ids = sentence_tokens['token_type_ids']
        attention_mask = sentence_tokens['attention_mask']

        try:
            start_padding_idx = (input_ids.squeeze(0) == self.pad_token_idx).nonzero()[0].data.cpu().numpy()[0]
        except IndexError:
            start_padding_idx = self.args.max_len

        mask = torch.zeros(start_padding_idx)

        mask_num = int(start_padding_idx * self.masking_ratio)
        random_list = [random.randint(0, start_padding_idx-2) for r in range(mask_num)]

        for idx in random_list:
            mask[idx] = 1.0

        mask_input = input_ids.squeeze(0)[:start_padding_idx]
        masked_inputs = mask_input.masked_fill(mask == 1, self.mask_token_idx)

        for idx in range(self.args.max_len - len(masked_inputs)):
            masked_inputs = torch.cat([masked_inputs, torch.tensor([self.pad_token_idx])], dim=-1)

        mask_labels = input_ids.squeeze(0).masked_fill(masked_inputs != self.mask_token_idx, -100)

        self.masked_input.append(masked_inputs)
        self.masked_labels.append(mask_labels)

        return torch.tensor(input_ids).squeeze(0), \
               torch.tensor(token_type_ids).squeeze(0), \
               torch.tensor(attention_mask).squeeze(0), \
               torch.tensor(int(label))

    def __getitem__(self, index):

        mask_data = {'masked_input': self.masked_input[index],
                     'masked_label': self.masked_labels[index]}

        input_data = {'input_ids': self.sentence[index],
                      'token_type_ids': self.token_type_ids[index],
                      'attention_mask': self.attention_mask[index],
                      'label': self.label[index].to(self.args.device)}

        input_data = self.metric.move2device(input_data, self.args.device)
        mask_data = self.metric.move2device(mask_data, self.args.device)

        return input_data, mask_data

    def __len__(self):
        return len(self.label)


# Get train, valid, test data loader and BERT tokenizer
def get_loader(args, metric):
    path_to_train_data = args.path_to_data + '/' + args.task + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.task + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.task + '/' + args.test_data

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if args.MLM == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer)
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=True)}
    else:
        if args.train == 'True' and args.test == 'False':
            train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer)
            valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer)
            train_iter.load_data('train')
            valid_iter.load_data('valid')

            loader = {'train': DataLoader(dataset=train_iter,
                                          batch_size=args.batch_size,
                                          shuffle=True),
                      'valid': DataLoader(dataset=valid_iter,
                                          batch_size=args.batch_size,
                                          shuffle=True)}

        elif args.train == 'False' and args.test == 'True':
            test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer)
            test_iter.load_data('test')

            loader = {'test': DataLoader(dataset=test_iter,
                                         batch_size=args.batch_size,
                                         shuffle=True)}

        elif args.train == 'True' and args.test == 'True':
            train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer)
            valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer)
            test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer)

            train_iter.load_data('train')
            valid_iter.load_data('valid')
            test_iter.load_data('test')
            loader = {'train': DataLoader(dataset=train_iter,
                                          batch_size=args.batch_size,
                                          shuffle=True),
                      'valid': DataLoader(dataset=valid_iter,
                                          batch_size=args.batch_size,
                                          shuffle=True),
                      'test': DataLoader(dataset=test_iter,
                                         batch_size=args.batch_size,
                                         shuffle=True)}
    return loader, tokenizer


if __name__ == '__main__':
    get_loader('test')
