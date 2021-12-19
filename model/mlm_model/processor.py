import os
import logging
from apex import amp
import torch.nn as nn
from tqdm import tqdm
import torch.quantization
import torch.optim as optim
from model.loss import Loss
from model.utils import Metric
from model.mlm_model.bert import BERT
from data.dataloader import get_loader

from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)
        self.lossf = Loss(args)
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss':  float('inf')}
        self.model_progress = {'loss': -1, 'iter': -1, 'acc': -1}

    def run(self, inputs, mask_inputs):

        if self.args.MLM == 'True':
            loss, acc = self.config['model'](inputs, mask_inputs)
        else:
            logits = self.config['model'](inputs, mask_inputs)

            loss = self.lossf.base(self.config, logits, inputs['label'])
            acc = self.metric.cal_acc(logits, inputs['label'])
        
        return loss, acc

    def progress(self, loss, acc):
        self.model_progress['loss'] += loss
        self.model_progress['iter'] += 1
        self.model_progress['acc'] += acc

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().numpy() / self.model_progress['iter']
        acc = self.model_progress['acc'].data.cpu().numpy() / self.model_progress['iter']

        return loss, acc

    def get_object(self, tokenizer, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(optim,
                                                    num_warmup_steps=self.args.warmup_ratio*train_total,
                                                    num_training_steps=train_total)

        return scheduler

    def model_setting(self):
        loader, tokenizer = get_loader(self.args, self.metric)

        model = BERT(self.args)
        model.to(self.args.device)

        criterion, optimizer = self.get_object(tokenizer, model)

        if self.args.train == 'True' and self.args.MLM != 'True':
            scheduler = self.get_scheduler(optimizer, loader['train'])
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'args': self.args,
                  'model': model}

        config = self.metric.move2device(config, self.args.device)

        if config['args'].fp16 == 'True':
            config['model'], config['optimizer'] = amp.initialize(
                config['model'], config['optimizer'], opt_level=config['args'].opt_level)

        if config['args'].MLM == 'True' and config['args'].Finetune == 'True':
            ckpt = config['args'].path_to_save + config['args'].ckpt
            config['model'].load_state_dict(torch.load(ckpt)['model'])
            for p in config['model'].parameters():
                p.requires_grad_(False)

        self.config = config

        return self.config

    def train(self):
        self.config['model'].train()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        for step, batch in enumerate(tqdm(self.config['loader']['train'])):
            self.config['optimizer'].zero_grad()

            inputs, mask_inputs = batch
            loss, acc = self.run(inputs, mask_inputs)

            if self.args.fp16 == 'True':
                with amp.scale_loss(loss, self.config['optimizer']) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.config['optimizer'].step()
            self.config['scheduler'].step()

            self.progress(loss, acc)

        return self.return_value()

    def valid(self):
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)
    
        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['valid']):

                inputs, mask_inputs = batch
                loss, acc = self.run(inputs, mask_inputs)

                self.progress(loss, acc)

        return self.return_value()

    def test(self):
        ckpt = self.config['args'].path_to_save + self.config['args'].ckpt
        self.config['model'].load_state_dict(torch.load(ckpt)['model'], strict=False)
        self.config['model'].eval()

        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['test']):

                inputs, mask_inputs = batch
                loss, acc = self.run(inputs, mask_inputs)

                self.progress(loss, acc)

        return self.return_value()

    def mlm(self):
        if self.args.Finetune == 'True':
            ckpt = self.config['args'].path_to_save + self.config['args'].ckpt
            self.config['model'].load_state_dict(torch.load(ckpt)['model'], strict=False)

        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['test']):

                inputs, mask_inputs = batch
                loss, acc = self.run(inputs, mask_inputs)

                self.progress(loss, acc)

        return self.return_value()
