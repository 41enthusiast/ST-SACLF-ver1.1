from dataset_processing.pacs import *
from model import AttnVGG
from pretrained_models import Vgg, Vgg16
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

class Classifier(pl.LightningModule):

    def __init__(self, lr,
                 gamma=2,
                 alpha=2,
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
        self.model = AttnVGG(self.hparams.num_classes,
                             Vgg16([2, 9, 22, 30], False),#Vgg([2, 9, 22, 30]),
                             self.hparams.dropout_type,
                             self.hparams.dropout_p)
        self.criterion = focal_loss(self.hparams.num_classes, self.hparams.gamma, self.hparams.alpha)
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

        loss = self.criterion(labels.squeeze(), preds) + reg_loss
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

        return {'val_loss': loss.detach(), 'val_accuracy': acc.mean().detach(), 'val_recall': recall.detach(),
                'val_precision': precision.detach(), 'val_f1': f1_val.detach()}

    def validation_epoch_end(self, step_outputs):
        print('Collecting val results')
        outputs = step_outputs

        averages = {}
        averages['val_loss'] = torch.stack([x['val_loss'].float() for x in outputs]).mean()
        averages['val_accuracy'] = torch.stack([x['val_accuracy'].float() for x in outputs]).mean()
        averages['val_recall'] = torch.stack([x['val_recall'].float() for x in outputs]).mean()
        averages['val_precision'] = torch.stack([x['val_precision'].float() for x in outputs]).mean()
        averages['val_f1'] = torch.stack([x['val_f1'].float() for x in outputs]).mean()

        global_val_table = wandb.Table(
            columns=['experiment name', 'loss', 'accuracy', 'recall', 'precision', 'f1 score'])
        global_val_table.add_data(EXPERIMENT_NAME, averages['val_loss'], averages['val_accuracy'],
                                   averages['val_recall'], averages['val_precision'], averages['val_f1'])
        self.logger.experiment.log({'Val table': global_val_table})
        self.log_dict(averages)

        return averages


    def test_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        loss, preds, acc = self.run_model(self.model, x, y)
        precision, recall = precision_recall(preds.argmax(dim=1), y, average='macro', num_classes=4)
        f1_val = f1_score(preds.argmax(dim=1), y, average='macro', num_classes=4)

        self.log_dict({'test_loss': loss.detach(), 'test_accuracy': acc.mean().detach(), 'test_recall': recall.detach(),
                       'test_precision': precision.detach(), 'test_f1': f1_val.detach()})

        return {'test_loss': loss.detach(), 'test_accuracy': acc.mean().detach(), 'test_recall': recall.detach(),
                'test_precision': precision.detach(), 'test_f1': f1_val.detach(), 'attention': batch}

    def test_epoch_end(self, step_outputs):
        print('Collecting test results')
        outputs = step_outputs

        averages = {}
        averages['test_loss'] = torch.stack([x['test_loss'].float() for x in outputs]).mean()
        averages['test_accuracy'] = torch.stack([x['test_accuracy'].float() for x in outputs]).mean()
        averages['test_recall'] = torch.stack([x['test_recall'].float() for x in outputs]).mean()
        averages['test_precision'] = torch.stack([x['test_precision'].float() for x in outputs]).mean()
        averages['test_f1'] = torch.stack([x['test_f1'].float() for x in outputs]).mean()

        global_test_table = wandb.Table(
            columns=['experiment name', 'loss', 'accuracy', 'recall', 'precision', 'f1 score'])
        global_test_table.add_data(EXPERIMENT_NAME, averages['test_loss'], averages['test_accuracy'],
                                   averages['test_recall'], averages['test_precision'], averages['test_f1'])
        self.logger.experiment.log({'Test table': global_test_table})
        self.log_dict(averages)

        # print('Test visualizations')

        # inputs = outputs[0]['attention']
        # #print(inputs.shape)
        
        # images = inputs[0]#inputs[0:16, :, :, :]
        # I = make_grid(images, nrow=4, normalize=True, scale_each=True)
        # _, c0, c1, c2, c3 = self.model(images)
        # print(I.shape, c0.shape, c1.shape, c2.shape, c3.shape)
        # attn0 = visualize_attn(I, c0)
        # attn1 = visualize_attn(I, c1)
        # attn2 = visualize_attn(I, c2)
        # attn3 = visualize_attn(I, c3)

        # viz_table = wandb.Table(
        #     columns=['image', 'layer 0', 'low layer', 'middle layer', 'end layer'])

        # w_img = wandb.Image(I)
        # w_attn0 = wandb.Image(attn0)
        # w_attn1 = wandb.Image(attn1)
        # w_attn2 = wandb.Image(attn2)
        # w_attn3 = wandb.Image(attn3)

        # viz_table.add_data(w_img, w_attn0, w_attn1, w_attn2, w_attn3)
        # self.logger.experiment.log({'Attention visualization': viz_table})

        # print('Making the confusion matrix')
        # cm = make_confusion_matrix(self.model, self.hparams.num_classes, test_loader, device)
        # cm_img = plot_confusion_matrix(cm, self.hparams.class_names)
        # w_cm = wandb.Image(cm_img)

        # # log most and least confident images
        # print('Logging the most and least confident images')
        # (lc_scores, lc_imgs), (mc_scores, mc_imgs) = get_most_and_least_confident_predictions(self.model,
        #                                                                                         test_loader, self.device)
        # w_lc = wandb.Image(make_grid(lc_imgs, nrow=4, normalize=True, scale_each=True))
        # w_mc = wandb.Image(make_grid(mc_imgs, nrow=4, normalize=True, scale_each=True))

        # self.logger.experiment.log({'Confusion Matrix': w_cm, 'Least Confident Images': w_lc, 'Most Confident Images': w_mc})

        return averages


def train_model(model_class, train_loader, val_loader, test_loader, epochs, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join('./checkpoints', model_class.__name__),
                         logger=wandb_logger,
                         gpus=1 if str(device) == "cuda:0" else 0,
                         accelerator='gpu', devices=1,
                         max_epochs=epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=0)
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


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATASET = 'kaokore'
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    EPOCHS = 20
    p1, p2 = [0.4, 0.5]#[0.58, 0.54]#[0.34, 0.31]##p1 for repr p2 for rare
    EXPERIMENT_NAME = f'nonorm-wtrain-allrando-st-kaokore-vgg16-p1c-{p1}-p2r-{p2}'
    dataset_root = '../..'
    #style_path = '../../visapp-data/fst-kaokore-ohem' #'data/kaokore-stylized'
    style_path = '../../visapp-data/fst-kaokore-ohem'

    transform_kaokore = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #normalization better overall, though it hurts the final acc
        ])

    transform_basic = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
            transforms.RandomCrop((256,256)) #this can be jank
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #normalization better overall, though it hurts the final acc
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
    else:
        
        train_ds = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/train', transform=transform_kaokore)
        #train_ds_wts = torch.ones(len(train_ds))*0.5
        # train_dataset_stylized_rare = ImageFolder(f'{style_path}-2-ub', transform=transform_kaokore)
        # train_dataset_stylized_rare = ImageFolder(f'{style_path}-rare', transform=transform_kaokore)
        # train_dataset_stylized_rare = stratified_split(train_dataset_stylized_rare, p2)
        # train_dataset_stylized_rep = ImageFolder(f'{style_path}-1-ub/fst-kaokore', transform=transform_kaokore)
        # train_dataset_stylized_rep = ImageFolder(f'{style_path}-representative', transform=transform_kaokore)
        # train_dataset_stylized_rep = stratified_split(train_dataset_stylized_rep, p1)
        # train_dataset = ConcatDataset([train_ds, train_dataset_stylized_rep, train_dataset_stylized_rep])
        # mixed_dataset = stratified_split(ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/train', transform=transform_basic), p1)
        print(p1,p2)
        mixed_dataset = stratified_split(ImageFolder(f'{dataset_root}/visapp-data/kaokore_control_v1', transform=transform_basic), [p2, p2, p1, p1])
        train_dataset = ConcatDataset([train_ds, mixed_dataset])
        #train_wts = torch.cat([train_ds_wts,train_dataset_stylized_rep_wts])#,train_dataset_stylized_rep_wts])

        val_dataset = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/dev', transform=transform_kaokore)
        test_dataset = ImageFolder(f'{dataset_root}/kaokore_imagenet_style/status/test', transform=transform_kaokore)

        print('Loading train')
        train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS,
                                    shuffle=True#sampler=WeightedRandomSampler(train_wts, len(train_wts))
                                    )
        print('Loading val')
        val_loader = DataLoader(val_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)
        print('Loading test')
        test_loader = DataLoader(test_dataset, batch_size= BATCH_SIZE, num_workers = NUM_WORKERS, shuffle=False)
        class_names = 'commoner  incarnation  noble  warrior'.split('  ')
    wandb_logger = WandbLogger(project = 'stcluster-classifier')

    print('starting train')
    classifier_model = train_model(Classifier,
                                  lr=1e-3,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  test_loader=test_loader,
                                  epochs=EPOCHS,

                                  gamma=2,#5 ,5 doesnt help
                                  alpha=2,
                                  dropout_type='dropout',
                                  dropout_p=0.5,#where did 0.2 come from
                                  num_classes=len(class_names),
                                  class_names = class_names,
                                  regularization_type='L1',
                                  weight_decay=1e-5,
                                  dataset_used = DATASET
                                  )
    print('Finished')
    