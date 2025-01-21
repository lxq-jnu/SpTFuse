#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import logging

logger = logging.getLogger()

class Optimizer(object):
    def __init__(self,
                lr0,
                model,
                momentum,
                wd,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power,
                it=0,
                *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = it
        self.optim = torch.optim.SGD(
                model.parameters(),
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)


    '''
    get_lr这段代码是一个学习率调度器的实现，根据当前迭代步数（self.it）和预定义的参数，计算当前的学习率（lr）值。具体的计算逻辑如下：
    如果当前迭代步数小于等于预热步数（self.warmup_steps），则使用预热学习率策略。
    预热学习率从 self.warmup_start_lr 开始，每经过一步迭代，学习率乘以 self.warmup_factor。
    如果当前迭代步数大于预热步数，则使用常规学习率策略。常规学习率通过计算一个衰减因子来逐渐减小学习率。
    衰减因子根据当前迭代步数与总迭代步数的比值进行计算，乘以 self.lr0 得到当前学习率。
    最终返回计算得到的学习率值。
    '''
    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()

