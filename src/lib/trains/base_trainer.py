from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter
import cv2
import os


class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    #import pdb
    #pdb.set_trace()
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)
    self.reconstruct_img = False
    self.accumulated_loss = 0 if self.opt.accumulated_grad else None

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def save_tensor_to_img(self, tensors, filenames, path):
    batch_size, channel, w, h = tensors.size()
    reconstruct_imgs = tensors.detach().cpu().numpy().transpose(0,2,3,1)
    for ind in range(batch_size):
      save_path = os.path.join(path, filenames[ind])
      cv2.imwrite(save_path, reconstruct_imgs[ind])
    pass

  def run_epoch(self, phase, epoch, data_loader, logger=None):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}

    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)    
      output, loss, loss_stats = model_with_loss(batch)
      if self.reconstruct_img:
        file_path = '/data/mry/code/CenterNet/debug_conflict_bt_class_recon'
        self.save_tensor_to_img(output['reconstruct_img'], batch['meta']['file_name'], file_path)
      loss = loss.mean()
      if phase == 'train':
        if opt.accumulated_grad:
          if iter_id==0:
            self.optimizer.zero_grad()
          loss.backward()
          self.accumulated_loss += loss.item()
        else:
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        if l == 'KL_loss':
          if loss_stats[l] is not None:
            avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l])
          else:
            avg_loss_stats[l].update(0, batch['input'].size(0))
          continue
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)


      if logger and iter_id % opt.logger_iteration == 0:
        logger.write_iteration(
          '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td))
        for l in avg_loss_stats:
          if loss_stats[l] is None:
            continue
          avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
          logger.write_iteration('|{} {:.4f} '.format(l, avg_loss_stats[l].avg))
          logger.scalar_summary('train_iteration_{}'.format(l), avg_loss_stats[l].avg, (epoch-1)*num_iters+iter_id)
        logger.write_iteration('\n')

      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    if phase == 'train' and opt.accumulated_grad:
      self.optimizer.step()
      ret['accumulated_loss'] = self.accumulated_loss / num_iters
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader, logger):
    return self.run_epoch('train', epoch, data_loader, logger=logger)