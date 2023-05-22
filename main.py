# encoding = utf-8
from numpy.lib.function_base import place
import torch
import torch.nn as nn
import numpy as np
import math
from Model import  CNN_base, CNN_stat, AttackNN, CNN_base1,AttackNN_blackbox, WhiteAttack, VGG_GrayScale, ResNet_GrayScale
import argparse
import json
import random 
import time
import torchvision
import torchvision.datasets as datasets 
import torch.nn.functional as F
import torchattacks
# from Attack import PGD
import copy
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
# from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torchvision.models as models

print("current pid:",os.getpid())

parser = argparse.ArgumentParser()


unloader = torchvision.transforms.ToPILImage()
# writer = SummaryWriter()

parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs')
parser.add_argument('--train', type=str, default='victim', help='Training the target model, incldues victim, shadow, attack, and test')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
parser.add_argument('--image', type=str, default='rgb', help='Image type, include RGB and Gray')
parser.add_argument('--dataset', type=str, default='cifar', help='Datasets that include "mnist, cifar, and Fashion')
parser.add_argument('--model_flag', type=str, default='vgg', help='Model that include "vgg and resnet')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
parser.add_argument('--ratio', type=float, default=1e-5, help='Learning rate for training')
parser.add_argument('--class_dim', type=int, default=10, help='The dimension of the class')
parser.add_argument('--mode', type=str, default='black', help='The model of the attack in our model, black, and white.')
parser.add_argument('--pretrain_num', type=int, default=15, help='The dimension of the class')
parser.add_argument('--clip', type=float, default=5.0, help='The max grad values') 
parser.add_argument('--load_exist', type=bool, default=False, help='Loading the existing trained parameters')
parser.add_argument('--save_model', type=str, default='./vgg_model.pth', help='Loading the existing trained parameters') 
parser.add_argument('--save_model_atk', type=str, default='./vgg_model_atk.pth', help='Loading the existing trained parameters') 
parser.add_argument('--save_optimizer', type=str, default='vgg_fashion_optimizer_model.pth', help='Loading the existing optimizer parameters') 
parser.add_argument('--drop_prob', type=float, default=0.5, help='The dropout rate in the model during the training')
parser.add_argument('--step', type=float, default=0, help='The dropout rate in the model during the training')
  
parser.add_argument('--learning_rate_decay_start', type=int, default=0, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
parser.add_argument('--learning_rate_decay_every', type=int, default=1, 
                    help='how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='how many iterations thereafter to drop LR?(in epoch)')


# args = parser.parse_args()
args = parser.parse_args(args=[])
params = vars(args)
print(json.dumps(params, indent = 2))

device = torch.device("cuda")  

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


 

def train(model, train_loader, args, train_losses, epoch, criterion): 
    model = model.to(device)
    parameters = model.parameters()
    # print(parameters
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
     
    train_losses = []
 
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data)
        # optimizer.zero_grad()
        if args.image =='rgb':
            data = data.to(device)/255.0
        else:
            data = data.to(device)
        target = target.to(device)
        output = model(data)
   
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    print("Saving parameters...")
    torch.save(model.state_dict(), args.save_model)
    print('\nTrain Epoch: {} \tLoss: {:.6f}'.format(epoch, np.mean(train_losses)))
 

    
    # torch.save(optimizer.state_dict(), args.save_optimizer)
 
def test(model, test_loader, args, criterion): 
    model.eval()
    test_losses = []
     
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            if args.image == 'rgb':
                data = data.to(device)/255.0
            else:
                data = data.to(device)
            target = target.to(device)
            output = model(data)

            test_loss = criterion(output, target).item()

            pred = output.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred)).sum()
 
            test_losses.append(test_loss)

    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
      np.mean(test_losses), correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
def static_value(model, i):
      
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11 = model.x1[i].item(), model.x2[i].item(), \
    model.x3[i].item(), model.x4[i].item(), model.x5[i].item(), model.x6[i].item(), model.x7[i].item(),\
    model.x8[i].item(), model.x9[i].item(), model.x10[i].item(), model.x11[i].item()
    return np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11])

def psnr(noise, image):
    image = np.array(image.cpu())
    noise = np.array(noise.detach().cpu())

    gap = image-noise

    mse = np.mean((gap)**2)
    if mse<1e-10:
        return 100
 
    return 10*math.log10(255.0**2/mse)
 
def none_zero(noise, image):
    image = np.array(image.cpu())
    noise = np.array(noise.detach().cpu())
    batch_size = image.shape[0]
    gap = image-noise 
    npzero = np.count_nonzero(gap, axis = 0)
    error = np.mean(gap**2)
    place_changed = np.sum(npzero)/batch_size
    return place_changed, error

def test_static(model, test_loader, atk, text_array): 
    model.eval()
    model = model.to(device)
 
    test_losses = []
    test_losses_atk = []
    test_loss_atk = 0
    test_loss = 0
    correct = 0
    diff_count_psnr = []
    diff_count_nonezero = []
    diff_mse = []
    
    correct_atk = 0
 
    counter = 0
    
    for data, target in test_loader:
        counter+=1
        with torch.no_grad():   
        
            if args.image == 'rgb':
                
                data = data.to(device)/255.0
            else:
                data = data.to(device)
            target = target.to(device) 
            output = model(data)
            loss =  F.nll_loss(output, target, size_average=False)

            test_loss = loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        data = data.to(device)
        tic = time.time()
        data_atk =  atk(data, target)#.to(device) 
        toc = time.time()
        place_changed, mse = none_zero(data_atk, data)
        psnr_i = psnr(data_atk, data)
        diff_count_psnr.append(psnr_i)
        diff_count_nonezero.append(place_changed)
        diff_mse.append(mse)
        output_atk = model(data_atk)
        test_losses.append(test_loss)


        
        loss_atk = F.nll_loss(output_atk, target, size_average=False)
        test_losses_atk.append(loss_atk.item())

        pred_atk = output_atk.data.max(1, keepdim=True)[1]

        correct_atk += pred_atk.eq(target.data.view_as(pred_atk)).sum()
        print("Total samples %d"%(counter*len(data)), "Correct %d"%correct_atk.item(),"Time %.4f"%(toc-tic))
    test_loss = np.mean(test_losses)
    test_loss_atk = np.mean(test_losses_atk)


    print('Test set normal: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset))) 

    print('Test set attack: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) PSNR: {:.4f} DotChanged {:.4f} MSE {:.4f}'.format(
      test_loss_atk, correct_atk, len(test_loader.dataset),
      100. * correct_atk / len(test_loader.dataset), np.mean(diff_count_psnr), np.mean(diff_count_nonezero), np.mean(diff_mse))) 
    # return text_array
 

def trainer(model, train_loader, test_loader, args):
    criterion_xen = nn.CrossEntropyLoss().to(device)
    # criterion_bce = nn.BCELoss().to(device)
    criterion_bce = nn.MultiLabelSoftMarginLoss().to(device)
    if args.load_exist: 
        checkpoint = torch.load(args.save_model, map_location=device) 

        print("Loading existing original parameters")
        model.load_state_dict(checkpoint) 
 


    train_losses = []
    train_counter = []
 
    for epoch in range(args.epochs): 
        train(model, train_loader, args, train_losses, epoch, criterion_xen)
        tic = time.time()
        test(model, test_loader, args, criterion_xen)
        toc = time.time()
        print("The time cost", toc-tic)


def tester(model, test_loader, args):
    model = model.to(device)
    model_atk = AttackNN(args).to(device)
    if os.path.exists(args.save_model): 
        checkpoint = torch.load(args.save_model, map_location=device)
        print("Loading existing parameters in test")
        model.load_state_dict(checkpoint)
    else:
        print("Please train model first!")
        
    
    
    if os.path.exists(args.save_model_atk):  
        checkpoint_atk = torch.load(args.save_model_atk, map_location=device)
        print("Loading existing attack parameters in test")
        model_atk.load_state_dict(checkpoint_atk)
        model_atk.apply(weights_init)
    else:
        print("Please train attacking model first!")

    # for param in model.parameters():
    #     param.requires_grad = False
    tic = time.time()
    # epss = np.arange(0, 1.0005,0.0005)
    text_array = []
    # for eps in epss:
 
    atk = torchattacks.FGSM(model, eps = 8/255)
    # atk = torchattacks.PGD(model_atk, eps=8/255, alpha=5/255, steps=20)
    # atk = torchattacks.TPGD(model_atk, eps=8/255, alpha=5/255, steps=20)
    # atk = torchattacks.GN(model_atk)
    # atk = torchattacks.APGD(model_atk, eps=8/255, steps=20)
    # atk = torchattacks.UPGD(model_atk, eps=8/255, alpha=5/255, steps=40, random_start=False, loss='ce', decay=1.0, eot_iter=1)
    # atk = torchattacks.CW(model_atk, c=1e-1, kappa=0, steps=1000, lr=0.05)
    # atk = torchattacks.MIFGSM(model_atk, eps=8/255, alpha=5/255, steps=20, decay=1.0)
    # atk = torchattacks.RFGSM(model_atk, eps=8/255, alpha=5/255, steps=20)
    # atk = torchattacks.OnePixel(model_atk, pixels=5, steps=10, popsize=200, inf_batch=32)
    # atk = torchattacks.SparseFool(model_atk, steps=10, lam=3, overshoot=0.2)
    # atk = torchattacks.AutoAttack(model_atk, norm='Linf', eps=8/255, version='standard', n_classes=10)
    # atk = torchattacks.BIM(model_atk,eps=8/255, alpha=5/255, steps=20)
    # atk = torchattacks.DeepFool(model_atk, steps=50, overshoot=0.02)
    # atk = torchattacks.FFGSM(model_atk, eps=8/255, alpha=5/255)
    # atk = torchattacks.PGDL2(model_atk, eps=8/255, alpha=5/255, steps=20, random_start=True, eps_for_division=1e-10)
    # atk = torchattacks.DIFGSM(model_atk, eps=8/255, alpha=5/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
    # model_atk = copy.deepcopy(model)
    # atk = PGD(model_atk, eps=8/255, alpha=5/255, steps=20)
    test_static(model, test_loader, atk, text_array)
    # test(model, test_loader)
    toc = time.time()   
    print("The time cost", toc-tic)
    # text_array = np.array(text_array).transpose()
            
    # np.savetxt('res.txt', text_array, delimiter=',',fmt='%.4f') 


 

def CWLoss(outputs, labels, criterion_xen, kappa=0.5):
    outputs = outputs.to(device)
    labels = labels.to(device)
    outputs_sf = F.softmax(outputs)
    index = len(outputs_sf[0])
    one_hot_labels = torch.eye(index).to(device)[labels]
    
    imean = torch.mean((1-one_hot_labels)*outputs_sf, dim=1)


    j = torch.masked_select(outputs_sf, one_hot_labels.bool())
    res =  j - imean


    res = torch.sum(res)
    return res

def attacker_train(model, model_atk, train_loader, args, train_losses, epoch, ratio):
     
    model = model.to(device).eval()
    model_atk = model_atk.to(device)
    if epoch<args.pretrain_num:
        parameters_trained = model_atk.parameters()
    else:
        parameters_trained = []
        for n, p in model_atk.named_parameters():
                if 'pert' in n: 
                    parameters_trained.append(p)
        # print(n)

    optimizer = torch.optim.Adam(parameters_trained, lr=args.learning_rate)


    MSELoss = nn.MSELoss()
    criterion_xen = nn.CrossEntropyLoss(reduce=False).to(device)
    criterion_bce = nn.BCELoss(reduction='sum')
    criterion_kld = nn.KLDivLoss()

    train_losses = []
    maxP = 0
    count = 0
     
    if epoch > args.learning_rate_decay_start and args.learning_rate_decay_start >= 0:
        frac = (epoch - args.learning_rate_decay_start) // args.learning_rate_decay_every
        decay_factor = args.learning_rate_decay_rate  ** frac
        ratio = ratio * decay_factor


    for batch_idx, (data, target) in enumerate(train_loader):
 
        data = data.to(device)
        if args.image == 'rgb':
            data_norm = data/255.0
        else:
            data_norm = data
        target = target.to(device)
        batch_size = data.size()[0]
        # print(data)
        # set_lr(optimizer, current_lr) 
        optimizer.zero_grad()
        output = model(data_norm)
        
        x_pert, x_pert_ori, rec, muvar = model_atk.perturb(data)  
        if args.mode == 'black':
            if epoch<args.pretrain_num:
                output_atk = model_atk(data) 
                # print(output_atk)
            else:
                output_atk = model_atk(data + x_pert)
        else:
            output_atk = model_atk(data + x_pert, model)            
 
        output_atk_sf = F.softmax(output_atk)
        output_atk_lsf = F.log_softmax(output_atk)
        output_sf = F.softmax(output)
        # print(output_atk)
        loss_imit = criterion_kld(output_sf, output_atk_sf)
        # print(target)
        mu, logvar =  muvar
        KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # loss_bce = criterion_bce(rec, data_norm)

        
        
        loss_cw = CWLoss(output_atk, target, criterion_xen)
        loss_cw_imit = criterion_xen(output_atk_sf, target).mean() 
        lossimit2 = criterion_xen(output_sf, target).mean() 
        loss_rec= ((rec - data_norm)**2).mean()
        loss_sigma = torch.norm(x_pert, p=2)
        # print(loss)
        loss_rec = KLD + loss_rec
        if args.mode == 'black':
            if epoch<args.pretrain_num:
                set_lr(optimizer, 3e-4)
                loss = loss_cw_imit + loss_imit + 1e-3*loss_rec 
            else:
                set_lr(optimizer, 3e-5) 
                
                # print(loss_cw.item(), loss_rec.item(), loss_imit.item(), loss_cw_imit.item())
                loss = loss_cw + loss_sigma + 1e-3*loss_rec # + 1 * loss_imit - 1.0 * loss_cw_imit #c #+ 1e-4*(KLD + loss_sigma)
                # print(loss_cw, loss_sigma)
         
        else:
            if epoch<args.pretrain_num:
                set_lr(optimizer, 3e-5)
                loss = loss_cw
            else:
                set_lr(optimizer, 4e-4) 
                
            
            # print("loading white attack")
                loss =  lossimit2 + loss_rec
        # if epoch>args.pretrain_num:    
        #     # adv_images_grad = data.clone()#.detach()
        #     # data.requires_grad = True  
        #     # loss_rec.allow_unused = True
        #     grad = torch.autograd.grad(loss_cw, rec, retain_graph=True, create_graph=False, allow_unused = True)[0]
        #     # print(grad)
        #     grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + 1e-6
        #     grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        #     loss_grad = MSELoss(grad, rec)
        #     loss = loss+loss_grad
        

        # if maxP < torch.max(model_atk.x_pert):
        if False:
            f, axarr = plt.subplots(1, 4)
            axarr[0].imshow(unloader(data[0].detach().squeeze()))
            # gt, _ = torch.max(target, dim=1)
            axarr[0].set_xlabel('GT %d'%target[0])

            axarr[1].imshow(unloader((rec[0]).detach().squeeze()))
            lab = output.data.max(1, keepdim=True)[1]
            axarr[1].set_xlabel('Normal Output %d'%lab[0])
            axarr[2].imshow(unloader(x_pert[0].detach().squeeze()))
            lab_atk = output_atk_sf.data.max(1, keepdim=True)[1]
            axarr[3].set_xlabel('Attacked Label %d'%lab_atk[0])

            axarr[3].imshow(unloader((x_pert + data)[0].detach().squeeze()))
            # pre_lab, _  = torch.max(pre_atk, dim=1)
            # axarr[3].set_xlabel('Attacked Label %d'%pre_lab[0])
            plt.show()
        if False:
            args.step += 1
            writer.add_scalar('KLD', KLD, args.step)
            writer.add_scalar('loss_bce', loss_bce, args.step)
            writer.add_scalar('loss_cw', loss_cw, args.step)
            writer.add_scalar('loss_sigma', loss_sigma, args.step)
            writer.add_scalar('loss', loss, args.step)
            writer.add_histogram('mu', mu, args.step)
            writer.add_histogram('var', logvar, args.step)
            writer.add_histogram('x_pert', x_pert, args.step)
            writer.add_histogram('PCNN_fc1', model_atk.fc1.weight, args.step)
            writer.add_histogram('PCNN_fc2', model_atk.fc2.weight, args.step)
            writer.add_histogram('enc1', model_atk.pert_att.enc1.weight, args.step)
            writer.add_histogram('enc2', model_atk.pert_att.enc2.weight, args.step)
            writer.add_histogram('enc3', model_atk.pert_att.enc3.weight, args.step)
            writer.add_histogram('enc4', model_atk.pert_att.enc4.weight, args.step)
     
            writer.add_histogram('dec1', model_atk.pert_att.dec1.weight, args.step)
            writer.add_histogram('dec2', model_atk.pert_att.dec2.weight, args.step)
            writer.add_histogram('dec3', model_atk.pert_att.dec3.weight, args.step)
            writer.add_histogram('dec4', model_atk.pert_att.dec4.weight, args.step)

        maxP = torch.max(x_pert)
        minP = torch.min(x_pert)
       

        loss.backward()
        optimizer.step()
 
 
        train_losses.append(loss.item())


    if epoch%10==0:
        torch.save(model_atk.state_dict(), args.save_model_atk)


 
    print('\nTrain Epoch: %d \tLoss: %.6f MaxP: %.1f MinP:%.1f Ratio: %e' %(epoch, np.mean(train_losses), maxP, minP, ratio))
 
def attacker_trainer(model, train_loader, test_loader, args):
 
    if os.path.exists(args.save_model): 
        checkpoint = torch.load(args.save_model, map_location=device)
        print("Loading existing parameters")
        model.load_state_dict(checkpoint)
    if args.mode == 'black':
        model_atk = AttackNN(args).to(device) 
    else:
        # modela = copy.deepcopy(model)
        model_atk = WhiteAttack(args).to(device)     


    if os.path.exists(args.save_model_atk):  
        print("Loading existing attack parameters")
        
        checkpoint_atk = torch.load(args.save_model_atk, map_location=device)
        
        
        model_atk.load_state_dict(checkpoint_atk)
        print("Weight initialized")
        model_atk.apply(weights_init)
    else:
        print("Please train the model first!")

    criterion = nn.CrossEntropyLoss().to(device)
    

    train_losses = []
    train_counter = []
    ratio = args.ratio
    for epoch in range(args.epochs): 
        attacker_train(model, model_atk, train_loader, args, train_losses, epoch, ratio)
        tic = time.time()
        test(model, test_loader, args, criterion)
        toc = time.time()
        attacker_test(model, model_atk, test_loader, args, epoch)
        tac = time.time()
        print("The time cost normal %.3f attack time %.3f"%((toc-tic),(tac-toc))) 
 

def attacker_test(model, model_atk, test_loader, args, epoch): 
    model.eval()
    model_atk.eval()
    test_losses = []
    test_loss = 0
    diff_count_psnr = []
    diff_count_nonezero = []
    diff_mse = []
    correct = 0
    correct_atk = 0
    count = 0

    with torch.no_grad():
        
 
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            count += 1
            if args.image == 'rgb':
                data_norm = data/255.0
            else:
                data_norm = data
            output_ori = model(data_norm)
            
            x_pert, x_pert_ori, rec, muvar  = model_atk.perturb(data)
            if args.mode == 'black':
                if epoch < args.pretrain_num:
                    output_atk = model_atk(data)
                else:
                    output_atk = model_atk(data+x_pert)
            else:
                output_atk = model_atk(data+x_pert, model)               
    
            data_atk = x_pert + data
            if args.image == 'rgb':
                data_atk_norm = data_atk/255.0
            else:
                data_atk_norm = data_atk
            output = model(data_atk_norm)
            
            place_changed, mse = none_zero(data_atk_norm, data_norm)
            psnr_i = psnr(data_atk_norm, data_norm)
            diff_count_psnr.append(psnr_i)
            diff_count_nonezero.append(place_changed)
            diff_mse.append(mse)

            test_loss += F.nll_loss(output, target, size_average=False).item()

            pred_ori = output_ori.data.max(1, keepdim=True)[1]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            pred_atk = output_atk.data.max(1, keepdim=True)[1]
            correct_atk += pred_atk.eq(target.data.view_as(pred_atk)).sum()

            if False:
                f, axarr = plt.subplots(1, 3)
                axarr[0].imshow(unloader(data[0].squeeze()))
                axarr[0].set_xlabel('True Label %d'%pred_ori[0])

                axarr[1].imshow(unloader(x_pert[0].detach().squeeze()))
                axarr[1].set_xlabel('Perturbation')

                axarr[2].imshow(unloader(data_atk[0].detach().squeeze()))
                axarr[2].set_xlabel('Attacked Label %d'%pred[0])

                plt.show()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('Test set: Avg. loss: {:.4f}, Accuracy: {:.2f}% Atk ACC: {:.2f}% PSNR: {:.4f} DotChanged {:.4f} MSE {:.4f}'.format(
      test_loss, 100. * correct / len(test_loader.dataset), 100. * correct_atk / len(test_loader.dataset), np.mean(diff_count_psnr), np.mean(diff_count_nonezero), np.mean(diff_mse)))

def weights_init(m): 
    classname=m.__class__.__name__
    if classname.find('pert'):
        for mi in list(m.children()):
            if isinstance(mi, nn.Conv2d):

                torch.nn.init.xavier_normal_(mi.weight.data)
                mi.bias.data.fill_(0)
            if isinstance(mi, nn.Linear):
                mi.weight.data.normal_()
                
def attacker_tester(model, test_loader, args):
    # model_atk = WhiteAttack(model)
    model_atk = AttackNN(args)
    model_atk = model_atk.to(device)

    if os.path.exists(args.save_model): 
        checkpoint = torch.load(args.save_model, map_location=device)
        print("Loading existing parameters")
        model.load_state_dict(checkpoint)
    else:
        print("Please train the model first!")
    if os.path.exists(args.save_model_atk): 
        checkpoint = torch.load(args.save_model_atk, map_location=device)
        print("Loading existing parameters")

        model_atk.load_state_dict(checkpoint)

    else:
        print("Please train the model first!")
    for param in model.parameters():
        param.requires_grad = False
    for param in model_atk.parameters():

        param.requires_grad = False
    # tic = time.time()
    # epss = np.arange(0, 1.0005,0.0005)
    # text_array = []
    # for eps in epss:
   
    attacker_test(model, model_atk, test_loader, args, 0)
    # test(model, test_loader)
    # toc = time.time()   
    # print("The time cost", toc-tic)
    # text_array = np.array(text_array).transpose()
class ToTensorWithoutScaling(object):
    """H x W x C -> C x H x W"""
    def __call__(self, image):
        return torch.ByteTensor(np.array(image)).permute(2, 0, 1).float()
if __name__=="__main__":
    print("==== Loading Data ...")
    
    if args.image == 'rgb':
        transform =  torchvision.transforms.Compose([
        ToTensorWithoutScaling() 
        ])
    else:
        transform =  torchvision.transforms.Compose([
         torchvision.transforms.ToTensor()
        ]) 
    if args.dataset == 'fashion':
        print("Dataset FashionMnist is loaded")
        train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./', train=True, download=True, transform= transform),
            batch_size=512, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./', train=False, download=True, transform= transform),
            batch_size=100, shuffle=False)
    elif args.dataset == 'mnist':
        print("Dataset Mnist is loaded")
        train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./', train=True, download=True,
                                    transform= transform),
            batch_size=512, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./', train=False, download=True,
                                    transform= transform),
            batch_size=100, shuffle=True)
    elif args.dataset == 'cifar':
        print("Dataset CIFAR10 is loaded")
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../UniADV/', train=True, download=True, transform= transform),
            batch_size=512, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../UniADV/', train=False, download=True, transform= transform),
            batch_size=100, shuffle=True)
    elif args.dataset == 'cifar100':
        print("Dataset CIFAR100 is loaded")
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./', train=True, download=True, transform= transform),
            batch_size=512, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./', train=False, download=True, transform= transform),
            batch_size=100, shuffle=True)
    else:
        print("Dataset is not included!")
     

    class_dims = 10
    if args.model_flag == 'vgg': 
        
        if args.image == 'rgb':
            print("Model vgg16 RGB is loaded")
            model = models.vgg16(pretrained=True).to(device)
 
            num_features = model.classifier[-1].in_features
            features = list(model.classifier.children())[:-1] # Remove last layer
            features.extend([nn.Linear(num_features, class_dims), nn.LogSoftmax(dim=1)]) # Add our layer with 4 outputs
            model.classifier = nn.Sequential(*features) # Replace the model classifier
        else:
            print("Model Vgg16 GrayScale is loaded")
            model = VGG_GrayScale(class_dims).to(device)
          
         
    elif args.model_flag == 'resnet':
#     print(model)
        if args.image =='rgb':
            print("Model Resnet RGB is loaded")
            model = models.resnet50(pretrained=True) 
    
            num_features = model.fc.in_features


            model.fc =  nn.Linear(num_features, class_dims)  # Add our layer with 4 outputs
        else:
            print("Model ResNet50 GrayScale is loaded")
            model = ResNet_GrayScale(class_dims).to(device)
    # print(model)
 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("The model parameters is", params)
    
    


    # for k in range(5):
    if args.train == "attack":
        attacker_trainer(model,  train_loader, test_loader, args)
    elif args.train =='test':
        attacker_tester(model, test_loader, args)
    elif args.train == "victim":
        trainer(model,  train_loader, test_loader, args)
        
    elif args.train == 'sota':
        tester(model, test_loader, args)
    


