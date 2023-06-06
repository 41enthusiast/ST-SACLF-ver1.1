
from model import AttnVGG, AttnResNet
from pretrained_models import VggN, ResNetN
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from utils import *
import time
from torchmetrics.functional import precision, recall, f1_score
from typing import List
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import ConcatDataset, WeightedRandomSampler

import optuna
from optuna.integration import PyTorchLightningPruningCallback

DATASET = 'kaokore'
BATCH_SIZE = 24
NUM_WORKERS = 8
EPOCHS = 10
EXPERIMENT_NAME = f'fullfinetune-kaokore-resnet50'
dataset_root = 'datasets'

class Classifier(pl.LightningModule):

    def __init__(self,
                 lr,
                 dropout_p,
                 weight_decay,
                 regularization_type,
                 dropout_type,
                 loss_fn='focal_loss',
                 num_classes=4,
                 class_names=[],
                 dataset_used='kaokore'
                 ):
        """
        Inputs
            
            lr - Learning rate of the outer loop Adam optimizer
            
        """
        super().__init__()
        print('Initializing model and train,val,test setup')
        self.save_hyperparameters()
        self.model = AttnResNet(self.hparams.num_classes,
                             ResNetN('resnet50','avgpool',
                                    ['conv1', 'layer2','layer3','layer4'],
                                    True),
                             self.hparams.dropout_type,
                             self.hparams.dropout_p)
        if self.hparams.loss_fn == 'focal_loss':
            self.criterion = focal_loss(self.hparams.num_classes, 2, 2) #gamma, alpha
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimzer, self.scheduler = self.configure_optimizers()
        
        #cache step outputs
        self.val_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
        return [optimizer], [scheduler]

    def run_model(self, model, imgs, labels):
        # Execute a model with given output layer weights and inputs
        outputs = model(imgs)
        if isinstance(outputs, list):
            preds = outputs[0]

        # compute the loss
        # regularization - specifically l1
        reg_loss = torch.tensor(0., requires_grad=True)
        if self.hparams.regularization_type == 'L1':
            for name, param in model.named_parameters():
                if 'weight' in name:
                    reg_loss = reg_loss + self.hparams.weight_decay * torch.norm(param, 1)
        else:
            reg_loss

        loss = self.criterion(preds, labels.squeeze()) + reg_loss
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return loss, preds, acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, _, acc = self.run_model(self.model, x, y)

        self.log('train_loss', loss.detach(), on_epoch=True)
        self.log('train_acc', acc.detach(), on_epoch=True)
        

        return {'loss': loss, 'acc': acc}  # Returning None means we skip the default training optimizer steps by PyTorch Lightning

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, preds, acc = self.run_model(self.model, x, y)

        self.log('val_loss', loss.detach(), on_epoch=True)
        self.log('val_acc', acc.detach(), on_epoch=True)

        precision_val= precision(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=self.hparams.num_classes)
        recall_val = recall(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=self.hparams.num_classes)
        f1_val = f1_score(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=self.hparams.num_classes)

        self.log_dict({'val_loss': loss.detach(),
                       'val_accuracy': acc.mean().detach(),
                       'val_recall': recall_val.detach(),
                       'val_precision': precision_val.detach(),
                       'val_f1': f1_val.detach()})

        self.val_step_outputs.append({'val_loss': loss.detach(),
                                      'val_accuracy': acc.mean().detach(),
                                      'val_y': y.detach(),
                                      'val_preds': preds.argmax(dim=1).detach()}
                                    )
        return self.val_step_outputs[-1]

    def test_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        loss, preds, acc = self.run_model(self.model, x, y)
        precision_val= precision(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=self.hparams.num_classes)
        recall_val = recall(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=self.hparams.num_classes)
        f1_val = f1_score(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=4)

        self.log_dict({'test_loss': loss.detach(),
                       'test_accuracy': acc.mean().detach(),
                       'test_recall': recall_val.detach(),
                       'test_precision': precision_val.detach(),
                       'test_f1': f1_val.detach()})

        self.test_step_outputs.append({'test_loss': loss.detach(),
                                      'test_accuracy': acc.mean().detach(),
                                      'test_y': y.detach(),
                                      'test_preds': preds.argmax(dim=1).detach(),
                                      'attention': batch}
                                    )
        return self.test_step_outputs[-1]



class MyCallback(pl.Callback):
    def __init__(self, testloader):
        super().__init__()
        self.testloader = testloader
        
    def on_validation_epoch_end(self, trainer, pl_module):
        print('Collecting val results')
        outputs = pl_module.val_step_outputs

        averages = {}
        averages['val_loss'] = torch.stack([x['val_loss'].float() for x in outputs]).mean()
        averages['val_accuracy'] = torch.stack([x['val_accuracy'].float() for x in outputs]).mean()
        
        val_y = torch.cat([x['val_y'].int() for x in outputs])
        val_preds = torch.cat([x['val_preds'].int() for x in outputs])
        

        precision_val= precision(val_preds, val_y, task='multiclass', average='macro', num_classes=pl_module.hparams.num_classes)
        recall_val = recall(val_preds, val_y, task='multiclass', average='macro', num_classes=pl_module.hparams.num_classes)
        f1_val = f1_score(val_preds, val_y, task='multiclass', average='macro', num_classes=pl_module.hparams.num_classes)

        averages['val_recall'] = recall_val.detach()
        averages['val_precision'] = precision_val.detach()
        averages['val_f1'] = f1_val.detach()

        global_val_table = wandb.Table(
            columns=['loss', 'accuracy', 'recall', 'precision', 'f1 score'])
        global_val_table.add_data(averages['val_loss'], averages['val_accuracy'],
                                   averages['val_recall'], averages['val_precision'], averages['val_f1'])
        trainer.logger.experiment.log({'Val table': global_val_table})
        pl_module.log_dict(averages)
        
        pl_module.val_step_outputs.clear()
        
    def on_test_epoch_end(self, trainer, pl_module):
        print('Collecting test results')
        outputs = pl_module.test_step_outputs

        averages = {}
        averages['test_loss'] = torch.stack([x['test_loss'].float() for x in outputs]).mean()
        averages['test_accuracy'] = torch.stack([x['test_accuracy'].float() for x in outputs]).mean()

        test_y = torch.cat([x['test_y'].int() for x in outputs])
        test_preds = torch.cat([x['test_preds'].int() for x in outputs])

        precision_val= precision(test_preds, test_y, task='multiclass', average='macro', num_classes=pl_module.hparams.num_classes)
        recall_val = recall(test_preds, test_y, task='multiclass', average='macro', num_classes=pl_module.hparams.num_classes)
        f1_val = f1_score(test_preds, test_y, task='multiclass', average='macro', num_classes=pl_module.hparams.num_classes)
        averages['test_recall'] = recall_val.detach()
        averages['test_precision'] = precision_val.detach()
        averages['test_f1'] = f1_val.detach()

        global_test_table = wandb.Table(
            columns=['loss', 'accuracy', 'recall', 'precision', 'f1 score'])
        global_test_table.add_data(averages['test_loss'], averages['test_accuracy'],
                                   averages['test_recall'], averages['test_precision'], averages['test_f1'])
        trainer.logger.experiment.log({'Test table': global_test_table})
        pl_module.log_dict(averages)

        print('Test visualizations')

        inputs = outputs[0]['attention']
        #print(inputs.shape)
        
        images = inputs[0]#inputs[0:16, :, :, :]
        I = make_grid(images, nrow=4, normalize=True, scale_each=True)
        _, c0, c1, c2, c3 = pl_module.model(images)
        print(I.shape, c0.shape, c1.shape, c2.shape, c3.shape)
        attn0 = visualize_attn(I, c0)
        attn1 = visualize_attn(I, c1)
        attn2 = visualize_attn(I, c2)
        attn3 = visualize_attn(I, c3)

        viz_table = wandb.Table(
            columns=['image', 'layer 0', 'low layer', 'middle layer', 'end layer'])

        w_img = wandb.Image(I)
        w_attn0 = wandb.Image(attn0)
        w_attn1 = wandb.Image(attn1)
        w_attn2 = wandb.Image(attn2)
        w_attn3 = wandb.Image(attn3)

        viz_table.add_data(w_img, w_attn0, w_attn1, w_attn2, w_attn3)
        trainer.logger.experiment.log({'Attention visualization': viz_table})

        print('Making the confusion matrix')
        cm = make_confusion_matrix(pl_module.model, pl_module.hparams.num_classes, test_loader, pl_module.device)
        cm_img = plot_confusion_matrix(cm, pl_module.hparams.class_names)
        w_cm = wandb.Image(cm_img)

        # log most and least confident images
        print('Logging the most and least confident images')
        (lc_scores, lc_imgs), (mc_scores, mc_imgs) = get_most_and_least_confident_predictions(pl_module.model,
                                                                                              test_loader,
                                                                                              pl_module.device,
                                                                                              pl_module.hparams.num_classes)
        w_lc = wandb.Image(make_grid(lc_imgs, nrow=4, normalize=True, scale_each=True))
        w_mc = wandb.Image(make_grid(mc_imgs, nrow=4, normalize=True, scale_each=True))

        trainer.logger.experiment.log({'Confusion Matrix': w_cm, 'Least Confident Images': w_lc, 'Most Confident Images': w_mc})

        pl_module.test_step_outputs.clear()
    
def train_model(trial, model_class, train_loader, val_loader, test_loader, epochs, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join('./checkpoints', model_class.__name__),
                         logger=wandb_logger,
                         accelerator='auto',
                         max_epochs=EPOCHS,
                         callbacks=[ModelCheckpoint(save_top_k = 10, save_weights_only=True, mode="max", monitor="val_acc", dirpath = './checkpoints'),
                                    LearningRateMonitor("epoch"),
                                   MyCallback(test_loader),
                                   PyTorchLightningPruningCallback(trial, monitor = 'val_acc')],
                         )
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        './checkpoints', 'onestagemodel' + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatloads the model with the saved hyperparameters
        model = model_class.load_from_checkpoint(pretrained_filename)
    
    pl.seed_everything(42)  # To be reproducable
    model = model_class(**kwargs)
    
    trainer.validating = True
    trainer.training = True
    
    trainer.fit(model, train_loader, val_loader)
    
    #print('Best model so far:', trainer.checkpoint_callback.best_model_path)

    return model, trainer.callback_metrics['val_acc'].item()

    # Training constant (same as for ProtoNet)
def get_dataset(trial):
    p1 = trial.suggest_float('p1', 0., 1.)
    p2 = trial.suggest_float('p2', 0., 1.)
    transform_kaokore = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            
        ])

    transform_basic = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
            transforms.RandomCrop((256,256)) 
            
        ])
        
    train_ds = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/train', transform=transform_kaokore)

    print(p1,p2)
    mixed_dataset = stratified_split(ImageFolder(f'{dataset_root}/kaokore_control_v1', transform=transform_kaokore), [p2, p2, p1, p1])
    train_dataset = ConcatDataset([train_ds, mixed_dataset])


    val_dataset = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/dev', transform=transform_kaokore)
    test_dataset = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/test', transform=transform_kaokore)

    print('Loading train')
    train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS,
                                shuffle=True
                                )
    print('Loading val')
    val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)
    print('Loading test')
    test_loader = DataLoader(test_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)

    return train_loader, val_loader, test_loader
    
def objective(trial):
    
    train_loader, val_loader, test_loader = get_dataset(trial)
    
    lr = trial.suggest_float('lr', 1e-8, 1e-1)
    dropout_type = trial.suggest_categorical('dropout_type', ['dropout', 'dropconnect'])
    dropout_p = trial.suggest_float('dropout_p', 0., 1.)
    reg_type = trial.suggest_categorical('reg_type', ['L1', 'L2'])
    wd = trial.suggest_float('wd', 1e-8, 1e-1)
    
    print('starting train')
    classifier_model, val_acc = train_model(trial, Classifier,
                                    train_loader,
                                    val_loader,
                                    test_loader,
                                    EPOCHS,

                                    lr=lr,
                                    loss_fn = 'focal_loss',
                                    dropout_type=dropout_type,
                                    dropout_p=dropout_p,
                                    num_classes=len(class_names),
                                    class_names = class_names,
                                    regularization_type= reg_type,
                                    weight_decay=wd,
                                    dataset_used = DATASET
                                  )
    return val_acc


if __name__ == '__main__':
    
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    class_names = 'commoner  incarnation  noble  warrior'.split('  ')
    wandb_logger = WandbLogger(project = 'stcluster-classifier')
    
    study = optuna.create_study(direction = 'maximize', pruner = pruner)
    study.optimize(objective, n_trials = 20, timeout = 600)
    
    print("Number of pruned trials: {}".format(len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])))
    print("Number of completed trials: {}".format(len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])))
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    print('Finished')
    
