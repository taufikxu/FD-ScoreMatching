import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation, anneal_dsm_score_estimation_TraceTrick
from losses.sliced_sm import anneal_sliced_score_estimation_vr
from losses.sliced_sm import anneal_ESM_scorenet, anneal_ESM_scorenet_VR
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
import time
__all__ = ['AnnealRunner']


class AnnealRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            if self.config.data.random_flip:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), download=False)
            else:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.ToTensor(),
                                 ]), download=False)

            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba_test'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=False)

        elif self.config.data.dataset == 'SVHN':
            dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                           transform=tran_transform)
            test_dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn_test'), split='test', download=True,
                                transform=test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        time_record = []
        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.


                if self.config.data.logit_transform:
                    X = self.logit_transform(X)


                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    t = time.time()
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'dsm_tracetrick':
                    t = time.time()
                    loss = anneal_dsm_score_estimation_TraceTrick(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    t = time.time()
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)
                elif self.config.training.algo == 'esm_scorenet':
                    t = time.time()
                    loss = anneal_ESM_scorenet(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)
                elif self.config.training.algo == 'esm_scorenet_VR':
                    t = time.time()
                    loss = anneal_ESM_scorenet_VR(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t = time.time() - t
                time_record.append(t)

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    tb_logger.add_scalar('loss', loss, global_step=step)
                    logging.info("step: {}, loss: {}, time per step: {:.3f} +- {:.3f} ms".format(step, loss.item(), 
                                                                                            np.mean(time_record) * 1e3, np.std(time_record) * 1e3))

                    # if step % 2000 == 0:
                    #     score.eval()
                    #     try:
                    #         test_X, test_y = next(test_iter)
                    #     except StopIteration:
                    #         test_iter = iter(test_loader)
                    #         test_X, test_y = next(test_iter)

                    #     test_X = test_X.to(self.config.device)
                    #     test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.


                    #     if self.config.data.logit_transform:
                    #         test_X = self.logit_transform(test_X)

                    #     test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    #     #if self.config.training.algo == 'dsm':
                    #     with torch.no_grad():
                    #         test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                    #                                                         self.config.training.anneal_power)

                    #     tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)
                    #     logging.info("step: {}, test dsm loss: {}".format(step, test_dsm_loss.item()))

                        # elif self.config.training.algo == 'ssm':
                        #     test_ssm_loss = anneal_sliced_score_estimation_vr(score, test_X, test_labels, sigmas,
                        #                                          n_particles=self.config.training.n_particles)

                        #     tb_logger.add_scalar('test_ssm_loss', test_ssm_loss, global_step=step)
                        #     logging.info("step: {}, test ssm loss: {}".format(step, test_ssm_loss.item()))



                if step >= 140000 and step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images


    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint_199000.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()
        grid_size = 5

        imgs = []
        if self.config.data.dataset == 'MNIST':
            samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)
            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))


        # else:
        #     samples = torch.rand(grid_size ** 2, 3, 32, 32, device=self.config.device)

        #     all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

        #     for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
        #         sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
        #                              self.config.data.image_size)

        #         if self.config.data.logit_transform:
        #             sample = torch.sigmoid(sample)

        #         image_grid = make_grid(sample, nrow=grid_size)
        #         if i % 10 == 0:
        #             im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
        #             imgs.append(im)

        #         save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)), nrow=10)
        #         torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        # imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)

        else:
            grid_size = 10
            for i in range(10):
                samples = torch.rand(grid_size ** 2, 3, 32, 32, device=self.config.device)
                all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)
                sample = all_samples[len(all_samples)-1].view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                         self.config.data.image_size)

                image_grid = make_grid(sample, nrow=grid_size)
                im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                im.save(os.path.join(self.args.image_folder, 'image_final_{}.png'.format(i)))




    def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):
        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 3, 32 ,32)
        half_refer_image = refer_image[..., :16]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :16] = corrupted_half_image
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :, :, :16] = corrupted_half_image
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def test_inpainting(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))
        score.eval()

        imgs = []
        if self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            refer_image, _ = next(iter(dataloader))

            samples = torch.rand(20, 20, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)
            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))

        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)
            elif self.config.data.dataset == 'SVHN':
                dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                               transform=transform)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)
            data_iter = iter(dataloader)
            refer_image, _ = next(data_iter)

            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))
            samples = torch.rand(20, 20, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size).to(self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))


        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)




### add by tianyu ###
    def anneal_Langevin_dynamics_GenerateImages(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):

        with torch.no_grad():
            for c, sigma in enumerate(sigmas):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return torch.clamp(x_mod, 0.0, 1.0).to('cpu')

    def save_sampled_images(self):
        num_of_step = 100
        bs = 100

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                        self.config.model.num_classes))

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        print('Load checkpoint from' + self.args.log)

        for epochs in [100000,170000]:
            states = torch.load(os.path.join(self.args.log, 'checkpoint_'+str(epochs)+'.pth'), map_location=self.config.device)
            score = CondRefineNetDilated(self.config).to(self.config.device)
            score = torch.nn.DataParallel(score)

            score.load_state_dict(states[0])
            score.eval()

            if not os.path.exists(os.path.join(self.args.image_folder, 'epochs'+str(epochs))):
                os.makedirs(os.path.join(self.args.image_folder, 'epochs'+str(epochs)))

            save_index = 0
            print("Begin epochs", epochs)
            for j in range(num_of_step):
                samples = torch.rand(bs, 3, 32, 32, device=self.config.device)
                all_samples = self.anneal_Langevin_dynamics_GenerateImages(samples, score, sigmas, 100, 0.00002)
                images_new = all_samples.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                for k in range(len(images_new)):
                    ims = Image.fromarray(images_new[k])
                    ims.save(os.path.join(self.args.image_folder, 'epochs'+str(epochs), 'img_'+str(save_index)+'.png'))
                    print('Save images ', k)
                    save_index+=1






    def calculate_fid(self):
        import fid
        import tensorflow as tf

        num_of_step = 500
        bs = 100

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                        self.config.model.num_classes))
        stats_path = 'fid_stats_cifar10_train.npz' # training set statistics
        inception_path = fid.check_or_download_inception(None) # download inception network

        print('Load checkpoint from' + self.args.log)
        #for epochs in range(140000, 200001, 1000):
        for epochs in [149000]:
            states = torch.load(os.path.join(self.args.log, 'checkpoint_'+str(epochs)+'.pth'), map_location=self.config.device)
            #states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
            score = CondRefineNetDilated(self.config).to(self.config.device)
            score = torch.nn.DataParallel(score)

            score.load_state_dict(states[0])
            

            score.eval()
            
            

            if self.config.data.dataset == 'MNIST':
                print("Begin epochs", epochs)
                samples = torch.rand(bs, 1, 28, 28, device=self.config.device)
                all_samples = self.anneal_Langevin_dynamics_GenerateImages(samples, score, sigmas, 100, 0.00002)
                images = all_samples.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
                for j in range(num_of_step-1):
                    samples = torch.rand(bs, 3, 32, 32, device=self.config.device)
                    all_samples = self.anneal_Langevin_dynamics_GenerateImages(samples, score, sigmas, 100, 0.00002)
                    images_new = all_samples.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
                    images = np.concatenate((images, images_new), axis=0)


            else:
                print("Begin epochs", epochs)
                samples = torch.rand(bs, 3, 32, 32, device=self.config.device)
                all_samples = self.anneal_Langevin_dynamics_GenerateImages(samples, score, sigmas, 100, 0.00002)
                images = all_samples.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
                for j in range(num_of_step-1):
                    samples = torch.rand(bs, 3, 32, 32, device=self.config.device)
                    all_samples = self.anneal_Langevin_dynamics_GenerateImages(samples, score, sigmas, 100, 0.00002)
                    images_new = all_samples.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
                    images = np.concatenate((images, images_new), axis=0)

            # load precalculated training set statistics
            f = np.load(stats_path)
            mu_real, sigma_real = f['mu'][:], f['sigma'][:]
            f.close()

            fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

            fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            print("FID: %s" % fid_value)

