import os, yaml, argparse, torch, math, time, random, datetime
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import ConfusionMatrix

from network.cylinder3d import Cylinder3D
from dataloader.semantickitti import SemanticKITTI
from utils.lovasz import lovasz_softmax
from utils.consistency_loss import PartialConsistencyLoss
from utils.evaluation import compute_iou
from pytorch_lightning import loggers as pl_loggers
import warnings
warnings.filterwarnings("ignore")

# 19 classes
class_name = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road',
              'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole',
              'traffic-sign']

class LightningTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        seed=123 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

        self.config = config
        self.save_hyperparameters(config)
        log_folder = config['logger']['root_dir']
        log_name = config['logger']['name']
        self.log_folder = f'{log_folder}/{log_name}'

        self._load_dataset_info()
        self.student = Cylinder3D(nclasses=self.nclasses, **config['model'])
        self.teacher = Cylinder3D(nclasses=self.nclasses, **config['model'])
        self.initialize_teacher()

        self.loss_ls = lovasz_softmax
        self.loss_cl = PartialConsistencyLoss(H=nn.CrossEntropyLoss, ignore_index=0)
        print('initialization', self.global_rank)

        self.teacher_cm = ConfusionMatrix(self.nclasses)
        self.student_cm = ConfusionMatrix(self.nclasses)
        self.best_miou = 0
        self.best_iou = np.zeros((self.nclasses-1,))

        self.train_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        self.val_dataset = SemanticKITTI(split='valid', config=self.config['dataset'])

        self.gpu_num = len(config['trainer']['devices'])
        self.train_batch_size = config['train_dataloader']['batch_size']
        if self.gpu_num > 1:
            self.effective_lr = math.sqrt((self.gpu_num)*(self.train_batch_size)) * self.config["optimizer"]["base_lr"]
        else:
            self.effective_lr = math.sqrt(self.train_batch_size) * self.config["optimizer"]["base_lr"]

        print(self.effective_lr)
        self.save_hyperparameters({'effective_lr': self.effective_lr})

        self.training_step_outputs = {'loss_list': [], 'first_step': None}
        self.eval_step_outputs = {'loss_list': [], 'hist_sum': None}

    def setup(self, stage=None):
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        seed = self.global_rank
       
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('local seed:', seed)
        ##############################################    

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.config['train_dataloader'],
                          collate_fn=self.collate_fn_BEV, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.config['val_dataloader'],
                          collate_fn=self.collate_fn_BEV, pin_memory=True)

    def forward(self, model, fea, pos, bs):
        # fea: list
        # pos: list
        # output_voxel = model([fea.squeeze(0)], [pos.squeeze(0)], bs)
        output_voxel = model(fea, pos, bs)  # [2,20,480,360,32]
        return output_voxel

    def training_step(self, batch, batch_idx):
        # if self.global_rank == 0:
        #     self.update_teacher()
        self.update_teacher()
        student_rpz_ten, student_fea_ten, student_label_ten = batch['student']
        teacher_rpz_ten, teacher_fea_ten, _ = batch['teacher']

        batch_size = len(student_fea_ten)
        student_output_voxel_batch = self(self.student, student_fea_ten, student_rpz_ten, batch_size)
        # student_output_batch[0].shape = [20, 119616]
        teacher_output_voxel_batch = self(self.teacher, teacher_fea_ten, teacher_rpz_ten, batch_size)

        # B = output_voxel.shape[0]     # pos [2,124xxx,3], len(pos)=2

        loss = torch.tensor([0.0], device=self.device)
        for i in np.arange(batch_size):
            student_label = student_label_ten[i].unsqueeze(0)

            student_output = student_output_voxel_batch[i, :, student_rpz_ten[i][:,0], student_rpz_ten[i][:,1], student_rpz_ten[i][:,2]].unsqueeze(0)
            teacher_output = teacher_output_voxel_batch[i, :, teacher_rpz_ten[i][:,0], teacher_rpz_ten[i][:,1], teacher_rpz_ten[i][:,2]].unsqueeze(0)
            loss += self.loss_cl(student_output, teacher_output, student_label) + \
               self.loss_ls(student_output.softmax(1), student_label, ignore=0)
            
        self.training_step_outputs['loss_list'].append(loss.cpu().detach())
        if self.training_step_outputs['first_step'] == None:
            self.training_step_outputs['first_step'] = self.global_step

        if self.global_rank == 0:
            self.log('train_loss_step', loss, prog_bar=True, rank_zero_only=True, logger=False) 

        return loss
    
    def on_train_epoch_end(self):
        loss_list = torch.stack(self.training_step_outputs['loss_list']).to(self.device)
        loss_list = self.all_gather(loss_list)
        if self.gpu_num > 1:
            loss_step_mean = loss_list.mean(dim=0)
        else:
            loss_step_mean = loss_list

        if self.global_rank == 0:
            first_step = self.training_step_outputs['first_step']
            for loss, step in zip(loss_step_mean, range(first_step, first_step+len(loss_step_mean))):
                self.logger.experiment.add_scalar('train_loss_step', loss, step)   
            
            self.logger.experiment.add_scalar('train_loss_epoch', loss_step_mean.mean(), self.current_epoch)

        del loss_list, loss_step_mean
        self.training_step_outputs['loss_list'].clear()
        self.training_step_outputs['first_step'] = None

    def validation_step(self, batch, batch_idx):
        student_rpz_ten, student_fea_ten, student_label_ten = batch['student']
        teacher_rpz_ten, teacher_fea_ten, teacher_label_ten = batch['teacher']

        batch_size = len(student_fea_ten)

        student_output_voxel_batch = self(self.student, student_fea_ten, student_rpz_ten, batch_size)
        teacher_output_voxel_batch = self(self.teacher, teacher_fea_ten, teacher_rpz_ten, batch_size)

        loss = torch.tensor([0.0], device=self.device)
        for i in np.arange(batch_size):
            teacher_label = teacher_label_ten[i].unsqueeze(0)
            student_label = student_label_ten[i].unsqueeze(0)

            student_output = student_output_voxel_batch[i, :, student_rpz_ten[i][:,0], student_rpz_ten[i][:,1], student_rpz_ten[i][:,2]].unsqueeze(0)
            teacher_output = teacher_output_voxel_batch[i, :, teacher_rpz_ten[i][:,0], teacher_rpz_ten[i][:,1], teacher_rpz_ten[i][:,2]].unsqueeze(0)

            mask = (teacher_label != 0).squeeze()     # student_label = teacher_label
            # print('update_cm', self.global_rank) 每个进程会经过4次此处 when BS=4
            # check below confusion matrix update with ddp, correct?
            self.student_cm.update(student_output.argmax(1)[:,mask], student_label[:,mask]) # 所有进程update sum叠加
            self.teacher_cm.update(teacher_output.argmax(1)[:,mask], teacher_label[:,mask])
            loss += self.loss_cl(student_output, teacher_output, student_label) + \
                   self.loss_ls(student_output.softmax(1), student_label, ignore=0)

        self.eval_step_outputs['loss_list'].append(loss.cpu().detach())

    def on_validation_epoch_end(self):
        loss_mean = torch.stack(self.eval_step_outputs['loss_list']).to(self.device).mean()
        loss_mean = self.all_gather(loss_mean).mean()

        if self.gpu_num > 1:
            student_cm_compute = self.all_gather(self.student_cm.compute().float()).mean(0)
            teacher_cm_compute = self.all_gather(self.teacher_cm.compute().float()).mean(0)
        else:
            student_cm_compute = self.student_cm.compute().float()
            teacher_cm_compute = self.teacher_cm.compute().float()

        if self.global_rank == 0:
            self.log('val_loss', loss_mean, prog_bar=True, logger=False)
            self.logger.experiment.add_scalar('val_loss', loss_mean, self.global_step) 

            student_iou, student_miou = compute_iou(student_cm_compute, ignore_zero=True)
            teacher_iou, teacher_miou = compute_iou(teacher_cm_compute, ignore_zero=True)

            if teacher_miou > self.best_miou:
                self.best_miou = teacher_miou
                self.best_iou = np.nan_to_num(teacher_iou) * 100
            self.log('val_best_miou', self.best_miou, on_epoch=True, prog_bar=True, logger=False)

            file_path = os.path.join(self.log_folder, 'val_iou.txt')
            file = open(file_path, "a")
            for classs, class_iou in zip(class_name, teacher_iou):
                print('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('teacher iou, training step:{0}'.format(self.global_step))
                file.write('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('\n')
            file.write('meanIOU={0}'.format(teacher_miou))
            file.write('\n')
            for classs, class_iou in zip(class_name, student_iou):
                print('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('student iou, training step:{0}'.format(self.global_step))
                file.write('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('\n')
            file.write('meanIOU={0}'.format(student_miou))
            file.write('\n')
            file.close()

        else:
            teacher_miou = torch.tensor(0).to(self.device)
            student_miou = torch.tensor(0).to(self.device)

        # reset ConfusionMatrix to 0 for all ranks
        self.student_cm.reset()
        self.teacher_cm.reset()

        if self.gpu_num > 1:
            teacher_miou = sum(self.all_gather(teacher_miou))
            student_miou = sum(self.all_gather(student_miou))

        self.log('val_teacher_miou', teacher_miou, prog_bar=True)
        self.log('val_student_miou', student_miou, prog_bar=True)

        del loss_mean
        self.eval_step_outputs['loss_list'].clear()
        
    def configure_optimizers(self):
        optimizer = Adam(self.student.parameters(), lr=self.effective_lr)
        return [optimizer]

    def initialize_teacher(self) -> None:
        self.alpha = 0.99 # TODO: Move to config
        for p in self.teacher.parameters(): p.detach_()

    def update_teacher(self) -> None:
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(alpha).add_(sp.data, alpha=1 - alpha)

    def _load_dataset_info(self) -> None:
        dataset_config = self.config['dataset']
        self.nclasses = len(dataset_config['labels'])
        self.unique_label = np.asarray(sorted(list(dataset_config['labels'].keys())))[1:] - 1
        self.unique_name = [dataset_config['labels'][x] for x in self.unique_label + 1]
        self.color_map = torch.zeros(self.nclasses, 3, device='cpu', requires_grad=False)
        for i in range(self.nclasses):
            self.color_map[i,:] = torch.tensor(dataset_config['color_map'][i][::-1], dtype=torch.float32)

    def collate_fn_BEV(self, data):
        stu_rpz2stack = []
        stu_fea2stack = []
        stu_label2stack = []
        tea_rpz2stack = []
        tea_fea2stack = []
        tea_label2stack = []

        for d in data:
            stu_rpz2stack.append(d['student'][0])
            stu_fea2stack.append(d['student'][1])
            stu_label2stack.append(d['student'][2])
            tea_rpz2stack.append(d['teacher'][0])
            tea_fea2stack.append(d['teacher'][1])
            tea_label2stack.append(d['teacher'][2])

        return {
                'student': (stu_rpz2stack, stu_fea2stack, stu_label2stack),
                'teacher': (tea_rpz2stack, tea_fea2stack, tea_label2stack)
            }


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/training.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))

    gpus = config['trainer']['devices']
    gpu_num = len(gpus)
    print(gpus)
    if gpu_num > 1:
        strategy='ddp'
        use_distributed_sampler = True
    else:
        strategy = 'auto'
        use_distributed_sampler = False

    log_folder = config['logger']['root_dir']
    log_name = config['logger']['name']
    os.makedirs(f'{log_folder}/{log_name}', exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_folder,
                                             name=log_name,
                                             default_hp_metric=False)
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{log_folder}/{log_name}', filename='{epoch}-{val_teacher_miou:.2f}',
                                                save_last=True, monitor='val_teacher_miou', mode='max', save_top_k=3)    
    
    backup_dir = os.path.join(log_folder, log_name, 'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp dataloader/semantickitti.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(args.config_path, backup_dir))

    model = LightningTrainer(config)

    Trainer(logger=tb_logger,
            callbacks=[checkpoint],
            strategy=strategy,
            use_distributed_sampler=use_distributed_sampler,
            log_every_n_steps=1,
            **config['trainer']).fit(model)