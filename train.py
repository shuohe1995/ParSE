import argparse
import torch
from utils.utils_data import prepare_cv_datasets, prepare_train_loaders_for_uniform_cv_candidate_labels
from utils.utils_algo import  AverageMeter,accuracy
from utils.utils_loss import partial_loss1,semantic_loss
from cifar_models.resnet import semantic_PLL
import numpy as np
from utils.cub200 import load_cub200
import math
import datetime
import os
from transformers import BertModel, BertTokenizer, logging
###################
###################
parser = argparse.ArgumentParser()
########
parser.add_argument('-batch_size', help='batch_size of ordinary labels.', default=128, type=int)
parser.add_argument('-dataset', help='specify a dataset', default='cifar100',choices=['cifar100','cifar100-H','cub'], type=str,required=False)
parser.add_argument('-n_class', help='the number of class', default=100,choices=[100,200], type=int)
parser.add_argument('-epoch', help='number of epochs', type=int, default=300)
parser.add_argument('-seed', help='Random seed', default=0, type=int, required=False)
parser.add_argument('-gpu', help='used gpu id', default='0', type=str, required=False)
parser.add_argument('-partial_rate', help='partial rate', default=0.01, type=float)
parser.add_argument('-exp_name', default='test', type=str)
#######
parser.add_argument('-lr', help='learning rate', default=1e-2, type=float)
parser.add_argument('-weight_decay', help='weight decay', default=1e-3, type=float)
parser.add_argument('-momentum', default=0.9, type=float, metavar='M',help='momentum of SGD solver')
parser.add_argument('-lr_decay_epochs', type=str, default='80,200',help='where to decay lr, can be a list')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,help='decay rate for learning rate')
parser.add_argument('-cosine', action='store_true', default=True,help='use cosine lr schedule')
######
parser.add_argument('-loss_weight', default=0.01, type=float)
parser.add_argument('-sigma', default=0.15, type=float)
parser.add_argument('-n_map', help='the number of layer in map', default=2, choices=[1,2,3], type=int)
parser.add_argument('-n_embedding', help='the dim of label embedding', default=768, type=int)
#####
args = parser.parse_args()
print(args)
####################
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
###################
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

def main():
    #######
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    #######
    ## log
    #######
    exp_path = args.dataset + '_' + str(args.partial_rate)
    exp_path = os.path.join('exp_log', exp_path)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    log_path=os.path.join(exp_path,str(datetime.datetime.now().year) + '-' + str(datetime.datetime.now().month) + '-' + str(datetime.datetime.now().day)+'-'+args.exp_name+'.txt')
    log = open(log_path, 'w')
    log_args(log)
    #####
    train_loader,test_loader, partial_Y,class_name=get_data_loader()
    class_embedding=get_class_embedding(class_name)
    ####
    tempY = partial_Y.sum(dim=1).unsqueeze(1).repeat(1, partial_Y.shape[1])
    init_confidence = partial_Y.float() / tempY
    init_confidence = init_confidence.to(device)
    ####
    loss_fn = partial_loss1(init_confidence)
    semantic_loss_fn=semantic_loss(args.sigma)
    model=get_model()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-4)

    avg_train_acc,avg_test_acc=train(log,model,optimizer,lr_scheduler,loss_fn,semantic_loss_fn,train_loader,test_loader,class_embedding)
    print("Average Test Accuracy over Last 10 Epochs:{:.4f}\n".format(avg_test_acc))
    print("Average Training Accuracy over Last 10 Epochs:{:.4f}\n".format(avg_train_acc))
    log.write("Average Test Accuracy over Last 10 Epochs:{:.4f}\n".format(avg_test_acc))
    log.write("Average Training Accuracy over Last 10 Epochs:{:.4f}\n".format(avg_train_acc))
    log.flush()
    return

def train(log,model,optimizer,lr_scheduler,loss_fn,semantic_loss_fn,train_loader,test_loader,class_embedding):
    train_acc_list = []
    test_acc_list = []
    for epoch in range(args.epoch):
        ######
        adjust_learning_rate(args, optimizer, epoch)
        train_one_epoch(log, epoch, model, optimizer, loss_fn,semantic_loss_fn, train_loader, train_acc_list,class_embedding)
        #lr_scheduler.step()
        test_one_epoch(log,epoch,model,test_loader,test_acc_list)
        ######
    avg_train_acc = np.mean(train_acc_list[-10:])
    avg_test_acc = np.mean(test_acc_list[-10:])
    return avg_train_acc, avg_test_acc

def train_one_epoch(log,epoch,model,optimizer,loss_fn,semantic_loss_fn,train_loader,train_acc_list,class_embedding):
    model.train()
    acc_train = AverageMeter('Acc@train', ':2.2f')
    acc_confidence = AverageMeter('Acc@confidence', ':2.2f')
    loss_cls = AverageMeter('loss@cls', ':2.2f')
    loss_cos = AverageMeter('loss@cos', ':2.2f')
    for i,(images1,images2,labels,true_labels,index) in enumerate(train_loader):
        X1,X2,Y,ty,index = images1.to(device),images2.to(device), labels.to(device),true_labels.to(device), index.to(device)
        outputs1,feature1 = model(X1)
        outputs2,feature2 = model(X2)
        outputs=torch.cat([outputs1,outputs2])
        loss1 = loss_fn(outputs,index)
        #######
        batch_confidence=torch.cat([loss_fn.confidence[index],loss_fn.confidence[index]])
        feature=torch.cat([feature1,feature2])
        loss2=semantic_loss_fn(feature,class_embedding,batch_confidence)
        loss = loss1 + args.loss_weight * loss2
        loss_fn.confidence_update(outputs1,outputs2,Y,index)
        ####
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #######
        acc = accuracy((outputs1+outputs2)/2,ty)[0]
        acc_train.update(acc[0])
        acc = accuracy(loss_fn.confidence[index], ty)[0]
        acc_confidence.update(acc[0])
        loss_cls.update(loss1.item())
        loss_cos.update(loss2.item())
        #########
    print('Train Epoch [{}]: lr:{:.4f}. Tr Acc:{:.2f}. Confidence Acc:{:.2f}.  loss_cls:{:.4f}. loss_alignment:{:.8f}'.format(epoch + 1, optimizer.param_groups[0]['lr'], acc_train.avg, acc_confidence.avg, loss_cls.avg,loss_cos.avg))
    log.write('Train Epoch [{}]: lr:{:.4f}. Tr Acc:{:.2f}. Confidence Acc:{:.2f}. loss_cls:{:.4f}. loss_alignment:{:.8f}\n'.format(epoch + 1, optimizer.param_groups[0]['lr'], acc_train.avg, acc_confidence.avg, loss_cls.avg,loss_cos.avg))
    log.flush()
    train_acc_list.extend([acc_train.avg.item()])

def test_one_epoch(log,epoch,model,test_loader,test_acc_list):
    model.eval()
    total, num_samples = 0, 0
    for i,(images, labels) in enumerate(test_loader):
        labels, images = labels.to(device), images.to(device)
        outputs,_ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
    test_acc= total / num_samples
    test_acc_list.extend([test_acc])
    print('Test  Epoch [{}]: Test Acc: {:.2%}.'.format(epoch + 1,test_acc))
    log.write('Test  Epoch [{}]: Test Acc: {:.2%}.\n'.format(epoch + 1, test_acc))
    log.flush()

def get_data_loader():
    if args.dataset == 'cub':
        image_size = 224
        train_loader, test_loader, partial_Y, class_name = load_cub200(image_size=image_size,partial_rate=args.partial_rate,batch_size=args.batch_size)
    else:
        train_data,train_labels, test_loader,class_name = prepare_cv_datasets(dataname=args.dataset, batch_size=args.batch_size)
        train_loader, partial_Y = prepare_train_loaders_for_uniform_cv_candidate_labels(dataname=args.dataset,partial_rate=args.partial_rate, data=train_data,labels=train_labels, batch_size=args.batch_size)
    return train_loader, test_loader, partial_Y,class_name

def get_model():
    is_pretrained=(args.dataset=='cub')
    model=semantic_PLL(pretrained=is_pretrained,n_class=args.n_class,n_embedding=args.n_embedding,n_map=args.n_map)
    model = model.to(device)
    return model

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epoch)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_class_embedding(class_name=[]):
    logging.set_verbosity_error()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    all_class_embedding = []
    for label in class_name:
        word_list=label.split(' ')
        label_embedding=torch.zeros(args.n_embedding)
        for word in word_list:
            text_dict = tokenizer.encode_plus(word, add_special_tokens=True, return_attention_mask=True)
            input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
            token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
            attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)
            res = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            words_vectors = res[0].detach().squeeze(0)
            wordvec = words_vectors[1]
            label_embedding +=wordvec.detach().clone()
        all_class_embedding.append(label_embedding)
    class_embedding=torch.stack(all_class_embedding).to(device)
    class_embedding = torch.nn.functional.normalize(class_embedding, dim=1).float()
    return class_embedding

def log_args(log):
    for key,value in vars(args).items():
        log.write(key+":{}\n".format(value))
    log.flush()

if __name__ == '__main__':
    main()


