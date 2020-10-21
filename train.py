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
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=10, type=int, help='train epoch number')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-3")
parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)")
parser.add_argument("--print_freq", default=1, type=int, help="the freq of print during training")
parser.add_argument("--save_freq", default=1, type=int, help="the freq of save checkpoint")

def main():
    opt = parser.parse_args()
    print(opt)
    # train_set = MyDataset('image_debug/trg_train', 'image_debug/src_train','train')
    # val_set = MyDataset('image_debug/trg_val', 'image_debug/src_val','val')
    train_set = MyDataset('image/trg_train', 'image/src_train','train')
    val_set = MyDataset('image/trg_val', 'image/src_val','val')
    # train_loader = DataLoader(dataset=train_set, batch_size=32, num_workers=0, shuffle=True)
    train_loader = DataLoader(dataset=train_set, batch_size=32, num_workers=4, shuffle=True)
    # val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        # netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
        netVGG.load_state_dict(torch.load('vgg_dict'))
        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])  # 去掉vgg19()的最后一层
                for param in self.feature.parameters():
                    param.requires_grad = False
            def forward(self, x):
                out = self.feature(x)
                return out
        netContent = _content_model()

    print("===> Setting GPU")
    # netG = Generator(opt.upscale_factor)
    netG = MyGenerator(opt.upscale_factor)
    gname = "MyG_3"
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = MyDiscriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    criterion = nn.MSELoss(reduction="sum")
    criterion_e = nn.L1Loss(reduction="sum")
    #view layer size
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vgg = netG.to(device)
    # summary(vgg, (1, 128, 161))
    #----------or-------------
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
        netD.cuda(1)
        criterion.cuda()
        criterion_e.cuda()
        if opt.vgg_loss:
            netContent.cuda(1)

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
    # #-------initialize--------
    # netG.load_state_dict(torch.load('checkpoint/checkpoint_G.pt')["netG"])
    # best_vec_loss = float('inf')
    # #-------------------------
    results = {'d_loss':[], 'image_loss': [], 'g_loss': [], 'ssim': [], 'vec_mse':[],'img_mse':[]}
    for epoch in range(opt.start_epoch + 1, opt.num_epochs + opt.start_epoch + 1):
        train_bar = tqdm(train_loader)
        # running_results = {'image_loss': 0, 'vgg_loss': 0}

        netG.train()
        netD.train()
        check = 0
        for data, target in train_bar:
            data, target = Variable(data), Variable(target, requires_grad=False)
            if opt.cuda:
                data = data.cuda()
                target = target.cuda(1)
            sr_img = netG(data)

            ############################
            # (1) Update D network:
            ###########################
            netD.zero_grad()
            real_out = netD(target)
            sr_img_t = sr_img.to(torch.device("cuda:1"))
            fake_out = netD(sr_img_t.detach())
            # fake_out = netD(sr_img_t)
            d_loss = criterion_e(real_out, fake_out)
            # d_loss.backward(retain_graph=True)
            d_loss.backward()
            optimizerD.step()

            ############################
            # (2) Update G network
            ###########################

            netG.zero_grad()
            image_loss = criterion(sr_img_t, target)
            if opt.vgg_loss:
                output_3channel = torch.stack([sr_img, sr_img, sr_img], 1).squeeze(2).to(torch.device("cuda:1"))
                target_3channel = torch.stack([target, target, target], 1).squeeze(2).to(torch.device("cuda:1"))
                content_input = netContent(output_3channel)
                content_target = netContent(target_3channel)
                content_target = content_target.detach()
                content_loss = criterion(content_input, content_target).to(torch.device("cuda:0"))
                netContent.zero_grad()
                # content_loss.backward(retain_graph=True)
                g_loss = image_loss + content_loss * 0.006 + d_loss
            # g_loss = torch.add(image_loss,d_loss_scalar)
            # else:alpha = 1
            alpha = 1
            # g_loss = image_loss+alpha*1/log10(d_loss.data)
            g_loss = 0.1*image_loss+d_loss.data*alpha
            # g_loss = image_loss
            g_loss.backward()
            optimizerG.step()
            # del data, target, sr_img,real_out,fake_out

            if opt.vgg_loss:
                train_bar.set_description(desc='[%d/%d] d_oss: %.2f g_loss: %.2f image_loss: %.2f vgg_loss: %.2f' % (
                    epoch, opt.num_epochs + opt.start_epoch, d_loss.item(), g_loss.item(), image_loss.item(), content_loss.item()))
            else:
                train_bar.set_description(desc='[%d/%d] alpha:%d d_loss: %.6f g_loss: %.6f image_loss: %.6f' % (
                    epoch, opt.num_epochs + opt.start_epoch, alpha, d_loss.item(), g_loss.item(), image_loss.item()))

            if check % 1000 == 0:  # and check != 0:
                image = []
                for num in range(5):
                    image.extend([display_transform()(data[num].cpu()),
                                  display_transform()(sr_img[num].cpu()),
                                  display_transform()(target[num].cpu())])
                image = torch.stack(image)
                # image = torch.chunk(image, image.size(0) // 15)
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, 'training_results/train/%s_batch_%d.png' % (gname,check), padding=5)
            check += 1

        netG.eval()
        netD.eval()
        out_path = 'training_results/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            valing_results = {'vec_mse': 0, 'img_mse': 0, 'ssim': 0}
            val_images = []
            val_bar = tqdm(val_loader)
            for j, batch in enumerate(val_bar):

                val_lr, val_hr = Variable(batch[0]), Variable(batch[1], requires_grad=False)
                if opt.cuda:
                    val_lr = val_lr.cuda()
                    val_hr = val_hr.cuda(1)
                val_sr = netG(val_lr)
                val_sr_t = val_sr.to(torch.device("cuda:1"))
                real_out = netD(val_hr)
                fake_out = netD(val_sr_t)
                vec_mse = criterion_e(real_out, fake_out)
                img_mse = criterion(val_hr, val_sr_t)

                valing_results['vec_mse'] += vec_mse.item()
                valing_results['img_mse'] += img_mse.item()
                # val_sr = val_sr.to(torch.device("cpu"))
                # val_hr = val_hr.to(torch.device("cpu"))
                batch_ssim = pytorch_ssim.ssim(val_sr_t, val_hr)
                valing_results['ssim'] += batch_ssim.item()
                train_bar.set_description(desc="validating……")
                # valing_results['psnr'] += 10 * log10(1 / valing_results['mse'])

                if j%2000==0:
                    val_images.extend(
                        [display_transform()(val_lr.data.cpu().squeeze(0)), display_transform()(val_sr.data.cpu().squeeze(0)),
                        display_transform()(val_hr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)

            index = 1
            for image in val_images:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + '%s_epoch_%d_index_%d.png' % (gname, epoch, index), padding=5)
                index += 1


        # save loss\scores\psnr\ssim\
        results['d_loss'].append(d_loss.item())
        results['image_loss'].append(image_loss.item())
        if opt.vgg_loss:
            results['vgg_loss'].append(content_loss.item())
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
            # save_checkpoint(netG, epoch)
            # torch.save(netG.state_dict(), 'checkpoint/model_epoch_%d.pth' % (epoch))
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