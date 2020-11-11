import argparse, os, shutil
import torch
import random
from math import log10
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from torchvision import models
# import torch.utils.model_zoo as model_zoo
from model import Generator,MyDiscriminator,MyGenerator
from dataset import DatasetFromFolder, MyDataset, display_transform
from tqdm import tqdm
# from torchsummary import summary
import pytorch_ssim
import csv
# import math
from torchvision.transforms import ToTensor, ToPILImage
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=10, type=int, help='train epoch number')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)")
parser.add_argument("--print_freq", default=1, type=int, help="the freq of print during training")
parser.add_argument("--save_freq", default=1, type=int, help="the freq of save checkpoint")

def main():
    opt = parser.parse_args()
    print(opt)
    train_set = MyDataset('image/trg_train', 'image/src_train','train')
    val_set = MyDataset('image/trg_val', 'image/src_val','val')
    train_loader = DataLoader(dataset=train_set, batch_size=32, num_workers=8, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=8, batch_size=32, shuffle=False)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    print("===> Setting GPU")
    netG = MyGenerator(opt.upscale_factor)
    gname = "MyG_3"
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = MyDiscriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    criterion = nn.MSELoss(reduction="sum")
    criterion_e = nn.L1Loss(reduction="sum")

    # --view layer size--
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg = netG.to(device)
    # summary(vgg, (1, 128, 161))
    #---------or---------
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #
    # # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])

    if opt.cuda:
        netG.cuda()
        netD.cuda()
        criterion.cuda()
        criterion_e.cuda()

    print("===> Setting Optimizer")
    optimizerG = optim.Adam(netG.parameters(),lr=opt.lr)
    optimizerD = optim.Adam(netD.parameters(),lr=opt.lr)

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            opt.start_epoch = checkpoint["epoch"]
            best_vec_loss = checkpoint["best_vec_loss"]
            netG.load_state_dict(checkpoint["netG"])
            netD.load_state_dict(checkpoint["netD"])
            optimizerG.load_state_dict(checkpoint["optimizerG"])
            optimizerD.load_state_dict(checkpoint["optimizerD"])
        else:
            print("=> no model found at '{}'".format(opt.pretrained))
    else:
        best_vec_loss = float('inf')
    #-------use pretrained netG--------
    # netG.load_state_dict(torch.load('checkpoint/checkpoint_G.pt')["netG"])
    #----------------------------------
    results = {'d_loss':[], 'image_loss': [], 'g_loss': [], 'ssim': [], 'vec_mse':[],'img_mse':[]}
    for epoch in range(opt.start_epoch + 1, opt.num_epochs + opt.start_epoch + 1):
        train_bar = tqdm(train_loader)

        netG.train()
        netD.train()

        for data, target in train_bar:
            data, target = Variable(data), Variable(target, requires_grad=False)
            if opt.cuda:
                data = data.cuda()
                target = target.cuda()
            sr_img = netG(data)

            ############################
            # (1) Update embedding block:
            ###########################
            netD.zero_grad()
            real_out = netD(target)
            fake_out = netD(sr_img.detach())
            d_loss = criterion_e(real_out, fake_out)
            d_loss.backward()
            optimizerD.step()

            ############################
            # (2) Update SR network
            ###########################
            netG.zero_grad()
            image_loss = criterion(sr_img, target)

            alpha = 1
            g_loss = 0.1*image_loss+d_loss.data*alpha
            g_loss.backward()
            optimizerG.step()

            train_bar.set_description(desc='[%d/%d] alpha:%d d_loss: %.6f g_loss: %.6f image_loss: %.6f' % (
                epoch, opt.num_epochs + opt.start_epoch, alpha, d_loss.item(), g_loss.item(), image_loss.item()))

        netG.eval()
        netD.eval()
        out_path = 'training_results/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            valing_results = {'vec_mse': 0, 'img_mse': 0, 'ssim': 0}
            val_bar = tqdm(val_loader)
            for j, batch in enumerate(val_bar):
                val_lr, val_hr = Variable(batch[0]), Variable(batch[1], requires_grad=False)
                if opt.cuda:
                    val_lr = val_lr.cuda()
                    val_hr = val_hr.cuda()
                val_sr = netG(val_lr)
                real_out = netD(val_hr)
                fake_out = netD(val_sr)
                vec_mse = criterion_e(real_out, fake_out)
                img_mse = criterion(val_hr, val_sr)

                valing_results['vec_mse'] += vec_mse.item()
                valing_results['img_mse'] += img_mse.item()
                # val_sr = val_sr.to(torch.device("cpu"))
                # val_hr = val_hr.to(torch.device("cpu"))
                batch_ssim = pytorch_ssim.ssim(val_sr, val_hr)
                valing_results['ssim'] += batch_ssim.item()
                train_bar.set_description(desc="validating……")

        # save loss\scores\psnr\ssim\
        results['d_loss'].append(d_loss.item())
        results['image_loss'].append(image_loss.item())
        results['g_loss'].append(g_loss.item())

        results['ssim'].append(valing_results['ssim']/len(val_loader))
        results['vec_mse'].append(valing_results['vec_mse']/len(val_loader))
        results['img_mse'].append(valing_results['img_mse']/len(val_loader))

        if epoch % opt.print_freq == 0:
            print("epoch: {}\tvector_loss:{} img_loss:{}".format(epoch, valing_results['vec_mse'] / len(val_loader),
                                                                 valing_results['img_mse'] / len(val_loader)))
        if epoch % opt.save_freq == 0:# and epoch != 0:
            vec_loss = valing_results['vec_mse']/len(val_loader)
            if vec_loss < best_vec_loss:
                best_vec_loss = vec_loss
                is_best = True
            else:
                is_best = False
            # print("Saving the model at iteration {} validation loss {}" \
            #       .format(epoch, vec_loss))
            save_checkpoint({
                "epoch": epoch,
                "best_vec_loss": best_vec_loss,
                "netG": netG.state_dict(),
                "netD": netD.state_dict(),
                "optimizerG": optimizerG.state_dict(),
                "optimizerD": optimizerD.state_dict()
            }, is_best, gname)

    with open('statistic/train_result_%s.csv' % gname,'a')as f:
        print(results)
        f_csv = csv.DictWriter(f, results.keys())
        f_csv.writeheader()
        f_csv.writerow(results)

def save_checkpoint(state, is_best, name):
    filename = "checkpoint/checkpoint_%s_epoch_%d.pt" % (name,state["epoch"])
    torch.save(state, filename)
    if is_best:
        print("saving the epoch {} as best model".format(state["epoch"]))
        shutil.copyfile(filename, 'checkpoint/bestmodel_%s.pt'%name)
    print("Checkpoint saved to {}".format(filename))

if __name__ == "__main__":
    main()