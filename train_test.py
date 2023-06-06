from dataset_processing.pacs import *
from model import AttnResNet
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
from torch.utils.data import ConcatDataset, WeightedRandomSampler

NUM_CLASSES = 8
LR = 0.00008
DROPOUT_P = 0.23
WD = 0.0004
DSTYPE = 'status-gender'
MODEL = 'resnet152'
DATASET = 'kaokore'
BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 20
p1, p2 = [0.8, 0.2]
EXPERIMENT_NAME = f'firststage-{DATASET}-{DSTYPE}-{MODEL}-p1c-{p1}-p2r-{p2}'
FFINETUNE = False
REG_TYPE = 'L2'

class Classifier(pl.LightningModule):

    def __init__(self, lr,
                 loss_fn='focal_loss',
                 dropout_type='dropout',
                 dropout_p=DROPOUT_P,
                 num_classes=NUM_CLASSES,
                 class_names=[],
                 regularization_type=REG_TYPE,
                 weight_decay=WD,
                 dataset_used=DATASET
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
                                    FFINETUNE),
                             self.hparams.dropout_type,
                             self.hparams.dropout_p)
        if self.hparams.loss_fn == 'focal_loss':
            self.criterion = focal_loss(self.hparams.num_classes, 2, 2) #gamma, alpha
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimzer, self.scheduler = self.configure_optimizers()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
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

        self.log_dict({'val_loss': loss.detach(), 'val_accuracy': acc.mean().detach(), 'val_recall': recall_val.detach(),
                       'val_precision': precision_val.detach(), 'val_f1': f1_val.detach()})

        return {'val_loss': loss.detach(), 'val_accuracy': acc.mean().detach(), 'val_y': y.detach(), 'val_preds': preds.argmax(dim=1).detach()}

    


    def test_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        loss, preds, acc = self.run_model(self.model, x, y)
        precision_val= precision(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=self.hparams.num_classes)
        recall_val = recall(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=self.hparams.num_classes)
        f1_val = f1_score(preds.argmax(dim=1), y, task='multiclass', average='macro', num_classes=self.hparams.num_classes)

        self.log_dict({'test_loss': loss.detach(), 'test_accuracy': acc.mean().detach(), 'test_recall': recall_val.detach(),
                       'test_precision': precision_val.detach(), 'test_f1': f1_val.detach()})

        return {'test_loss': loss.detach(), 'test_accuracy': acc.mean().detach(), 'test_y': y.detach(), 'test_preds': preds.argmax(dim=1).detach(), 'attention': batch}



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
        self.logger.experiment.log({'Test table': global_test_table})
        self.log_dict(averages)

        print('Test visualizations')

        inputs = outputs[0]['attention']
        #print(inputs.shape)
        
        images = inputs[0]#inputs[0:16, :, :, :]
        I = make_grid(images, nrow=4, normalize=True, scale_each=True)
        _, c0, c1, c2, c3 = self.model(images)
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
        self.logger.experiment.log({'Attention visualization': viz_table})

        print('Making the confusion matrix')
        cm = make_confusion_matrix(self.model, self.hparams.num_classes, test_loader, device)
        cm_img = plot_confusion_matrix(cm, self.hparams.class_names)
        w_cm = wandb.Image(cm_img)

        # log most and least confident images
        print('Logging the most and least confident images')
        (lc_scores, lc_imgs), (mc_scores, mc_imgs) = get_most_and_least_confident_predictions(self.model,
                                                                                                test_loader, self.device, self.hparams.num_classes)
        w_lc = wandb.Image(make_grid(lc_imgs, nrow=4, normalize=True, scale_each=True))
        w_mc = wandb.Image(make_grid(mc_imgs, nrow=4, normalize=True, scale_each=True))

        pl_module.log_dict(averages)
        pl_module.logger.experiment.log({'Confusion Matrix': w_cm, 'Least Confident Images': w_lc, 'Most Confident Images': w_mc})

        pl_module.test_step_outputs.clear()

def train_model(model_class, train_loader, val_loader, test_loader, epochs, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join('./checkpoints', model_class.__name__),
                         logger=wandb_logger,
                         
                         accelerator='auto',
                         max_epochs=epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         )
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        './checkpoints', "onstagemodel_resnet50.ckpt")
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


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    dataset_root = 'datasets'

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
    elif DATASET == 'dtd':
        class_names = os.listdir(f'{dataset_root}/imagenetstyle/train')
    else:
        if DSTYPE == 'status':
            srcfolname = 'kaokore_imagenet_style/status'
            stylefolname = 'kaokore_control_v1'
            strat_prob = [p2, p2, p1, p1]
        else:
            srcfolname = 'kaokore_imagenet_style_8way'
            stylefolname = 'kaokore-8way-stylized'
            strat_prob = [p1, p2, p1, p2, p2, p1, p1, p2]
        class_names = os.listdir(f'{dataset_root}/{srcfolname}/train') #for confusion matrix only, listdir holds
        train_ds = ImageFolder(f'{dataset_root}/{srcfolname}/train', transform=transform_kaokore)
        
        print(p1,p2)
        mixed_dataset = stratified_split(ImageFolder(f'{dataset_root}/{stylefolname}', transform=transform_kaokore), strat_prob)
        train_dataset = ConcatDataset([train_ds, mixed_dataset])
        

        val_dataset = ImageFolder(f'{dataset_root}/{srcfolname}/dev', transform=transform_kaokore)
        test_dataset = ImageFolder(f'{dataset_root}/{srcfolname}/test', transform=transform_kaokore)

        print('Loading train')
        train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS,
                                    shuffle=True
                                    )
        print('Loading val')
        val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)
        print('Loading test')
        test_loader = DataLoader(test_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)
        
    wandb_logger = WandbLogger(project = 'stclassifier')

    print('starting train')
    classifier_model = train_model(Classifier,
                                    lr=LR,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    test_loader=test_loader,
                                    epochs=EPOCHS,

                                    loss_fn = 'focal_loss',
                                    dropout_type='dropout',
                                    dropout_p=DROPOUT_P,
                                    num_classes=len(class_names),
                                    class_names = class_names,
                                    regularization_type= 'L2',
                                    weight_decay=WD,
                                    dataset_used = DATASET
                                  )
    print('Finished')
    
