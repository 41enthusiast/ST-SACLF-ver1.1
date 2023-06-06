import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from dataset_processing.pacs import *
from model import AttnResNet
from pretrained_models import ResNetN
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from utils import *
import time
from torchmetrics.functional import precision_recall, f1_score
from typing import List
from torchvision.utils import make_grid
from torch.utils.data import ConcatDataset, WeightedRandomSampler
import gc
from torch import nn


class Classifier(pl.LightningModule):

    def __init__(self, lr,
                 loss_fn='focal_loss',
                 dropout_type='dropout',
                 dropout_p=0.2,
                 num_classes=4,
                 class_names=[],
                 regularization_type='L1',
                 weight_decay=1e-6,
                 dataset_used='kaokore'
                 ):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            num_inner_steps - Number of inner loop updates to perform
        """
        super().__init__()
        print('Initializing model and train,val,test setup')
        self.save_hyperparameters()
        self.model = AttnResNet(self.hparams.num_classes,
                             ResNetN('resnet34','avgpool',
                              ['conv1', 'layer2','layer3','layer4']),
                             self.hparams.dropout_type,
                             self.hparams.dropout_p)
        if self.hparams.loss_fn == 'focal_loss':
            self.criterion = focal_loss(self.hparams.num_classes, 2, 2) #gamma, alpha
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimzer, self.scheduler = self.configure_optimizers()

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

        precision, recall = precision_recall(preds.argmax(dim=1), y, average='macro', num_classes=self.hparams.num_classes)
        f1_val = f1_score(preds.argmax(dim=1), y, average='macro', num_classes=self.hparams.num_classes)

        self.log_dict({'val_loss': loss.detach(), 'val_accuracy': acc.mean().detach(), 'val_recall': recall.detach(),
                       'val_precision': precision.detach(), 'val_f1': f1_val.detach()})

        return {'val_loss': loss.detach(), 'val_accuracy': acc.mean().detach(), 'val_y': y, 'val_preds': preds.argmax(dim=1)}

    def validation_epoch_end(self, step_outputs):
        print('Collecting val results')
        outputs = step_outputs

        averages = {}
        averages['val_loss'] = torch.stack([x['val_loss'].float() for x in outputs]).mean()
        averages['val_accuracy'] = torch.stack([x['val_accuracy'].float() for x in outputs]).mean()
        
        val_y = torch.cat([x['val_y'].int() for x in outputs])
        val_preds = torch.cat([x['val_preds'].int() for x in outputs])
        

        precision, recall = precision_recall(val_preds, val_y, average='macro', num_classes=self.hparams.num_classes)
        f1_val = f1_score(val_preds, val_y, average='macro', num_classes=self.hparams.num_classes)

        averages['val_recall'] = recall.detach()
        averages['val_precision'] = precision.detach()
        averages['val_f1'] = f1_val.detach()

        global_val_table = wandb.Table(
            columns=['loss', 'accuracy', 'recall', 'precision', 'f1 score'])
        global_val_table.add_data(averages['val_loss'], averages['val_accuracy'],
                                   averages['val_recall'], averages['val_precision'], averages['val_f1'])
        self.logger.experiment.log({'Val table': global_val_table})
        self.log_dict(averages)

        #return averages


    def test_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        loss, preds, acc = self.run_model(self.model, x, y)
        precision, recall = precision_recall(preds.argmax(dim=1), y, average='macro', num_classes=4)
        f1_val = f1_score(preds.argmax(dim=1), y, average='macro', num_classes=4)

        self.log_dict({'test_loss': loss.detach(), 'test_accuracy': acc.mean().detach(), 'test_recall': recall.detach(),
                       'test_precision': precision.detach(), 'test_f1': f1_val.detach()})

        return {'test_loss': loss.detach(), 'test_accuracy': acc.mean().detach(), 'test_y': y, 'test_preds': preds.argmax(dim=1), 'attention': batch}

    def test_epoch_end(self, step_outputs):
        print('Collecting test results')
        outputs = step_outputs

        averages = {}
        averages['test_loss'] = torch.stack([x['test_loss'].float() for x in outputs]).mean()
        averages['test_accuracy'] = torch.stack([x['test_accuracy'].float() for x in outputs]).mean()

        test_y = torch.cat([x['test_y'].int() for x in outputs])
        test_preds = torch.cat([x['test_preds'].int() for x in outputs])

        precision, recall = precision_recall(test_preds, test_y, average='macro', num_classes=self.hparams.num_classes)
        f1_val = f1_score(test_preds, test_y, average='macro', num_classes=self.hparams.num_classes)
        averages['test_recall'] = recall.detach()
        averages['test_precision'] = precision.detach()
        averages['test_f1'] = f1_val.detach()

        global_test_table = wandb.Table(
            columns=['loss', 'accuracy', 'recall', 'precision', 'f1 score'])
        global_test_table.add_data(averages['test_loss'], averages['test_accuracy'],
                                   averages['test_recall'], averages['test_precision'], averages['test_f1'])
        self.logger.experiment.log({'Test table': global_test_table})
        self.log_dict(averages)

        #return averages


def train_model(model_class, train_loader, val_loader, test_loader, epochs, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join('./checkpoints', model_class.__name__),
                         logger=wandb_logger,
                         gpus=1 if str(device) == "cuda:0" else 0,
                         accelerator='gpu', devices=1,
                         max_epochs=epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         )
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        './checkpoints', model_class.__name__ + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = model_class(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = model_class.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training
        trainer.test(model,test_loader)

    return model

    # Training constant (same as for ProtoNet)

def train(config = None):

    transform_kaokore = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            ])

    # transform_basic = transforms.Compose([
    #         transforms.Resize((256,256)),
    #         transforms.ToTensor(),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
    #         transforms.RandomCrop((256,256)) #this can be jank
    #     ])
    with wandb.init(config=config, project = 'stcluster-classifier-sweep'):
        config = wandb.config
        if DATASET == 'pacs':
            label_domain = os.listdir(f'{dataset_root}/pacs_data')[0]
            EXPERIMENT_NAME = 'PACS_indv_domain_train_'+label_domain
            print('Loading train')
            train_loader, _ = get_domain_dl(label_domain)
            print('Loading val')
            val_loader, _ = get_domain_dl(label_domain, 'crossval')
            print('Loading test')
            test_loader, _ = get_domain_dl(label_domain, 'test')
            class_names = 'dog  elephant  giraffe  guitar  horse  house  person'.split('  ')
        else:
            
            train_ds = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/train', transform=transform_kaokore)
            print(float(config.p1), float(config.p2))
            mixed_dataset = stratified_split(ImageFolder(f'{dataset_root}/kaokore_control_v1', transform=transform_kaokore), [float(config.p2), float(config.p2), float(config.p1), float(config.p1)])
            train_dataset = ConcatDataset([train_ds, mixed_dataset])

            val_dataset = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/dev', transform=transform_kaokore)
            test_dataset = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/test', transform=transform_kaokore)

            print('Loading train')
            train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS,
                                        shuffle=True)
            print('Loading val')
            val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)
            print('Loading test')
            test_loader = DataLoader(test_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)
            class_names = 'commoner  incarnation  noble  warrior'.split('  ')
        

        print('starting train')

        gc.collect()
        torch.cuda.empty_cache()

        classifier_model = train_model(Classifier,
                                    lr=0.0008,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    test_loader=test_loader,
                                    epochs=EPOCHS,

                                    loss_fn = 'focal_loss',
                                    dropout_type='dropout',
                                    dropout_p=0.23,
                                    num_classes=len(class_names),
                                    class_names = class_names,
                                    regularization_type= 'L2',
                                    weight_decay=0.0004,
                                    dataset_used = DATASET
                                    )
    gc.collect()
    torch.cuda.empty_cache()
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATASET = 'kaokore'
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    EPOCHS = 20
    #EXPERIMENT_NAME = f'hyperparam-sweep-kaokore-vgg16-p1c-{p1}-p2r-{p2}'
    dataset_root = '..'

    wandb_logger = WandbLogger(project = 'stcluster-classifier-sweep')
    
    

    train()

    print('Finished')

