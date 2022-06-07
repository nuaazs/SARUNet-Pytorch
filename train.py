# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝

# @Time    : 2021-09-28 09:35:49
# @Author  : zhaosheng
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : iint.icu
# @File    : train.py
# @Describe: training script for MR-to-CT translation

import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.test_options import TestOptions
import torch
import numpy as np
import random
from util.visualizer import save_images
from collections import OrderedDict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def traink(model, visualizer, train_loader, val_loader, opt):
    new_k_index = True
    test_scores = {'X': [], 'Y': [], 'legend': [
        f"K:{opt.k_index} Val Score", f"K:{opt.k_index} Test Score"]}
    lr_dict = {'X': [], 'Y': [], 'legend': [f"K:{opt.k_index} Learning Rate"]}
    """k-fold training

    Args:
        model ([type]): model to be trained.
        visualizer ([type]): [description]
        train_loader ([type]): train set dataloader.
        val_loader ([type]): valid set dataloader.
        opt ([type]): other options.

    Returns:
        [type]: [description]
    """
    setup_seed(123)
    total_iters = 0
    best_score = 9999
    best_epoch = 0
    name = opt.name

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        opt.now_epoch = epoch
        train_loss_list = []

        lr = model.optimizers[0].param_groups[0]['lr']
        lrs = OrderedDict([(f"K:{opt.k_index} Learning Rate", lr)])
        lr_dict['X'].append(epoch)
        lr_dict['Y'].append([lrs[k] for k in lr_dict['legend']])

        lr_X = np.stack([np.array(lr_dict['X'])] *
                        len(lr_dict['legend']), 1).flatten()
        lr_Y = np.array(lr_dict['Y']).flatten()

        if opt.display_id > 0:
            visualizer.vis.line(
                X=lr_X,
                Y=lr_Y,
                opts={
                    'title': opt.name + f"K:{opt.k_index} Learning Rate",
                    'legend': lr_dict['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'learning rate'},
                win=444+opt.k_index)

        print(f"learning rate {lr:.7f}")
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0
        # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        visualizer.reset()
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.

        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            # unpack data from dataset and apply preprocessing
            model.set_input(data)
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                if opt.display_id > 0:
                    visualizer.display_current_results(
                        model.get_current_visuals(), epoch, save_result, 888, f"Train - {opt.name}")

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses(opt.k_index)
                # print(losses)
                train_loss_list.append(losses[f"K:{str(opt.k_index)} G_L1"])
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(
                        epoch_iter) / len(train_loader), losses, new_k_index)
                    new_k_index = False

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' %
                      (epoch, total_iters))
                save_suffix = 'k%d_iter_%d' % (
                    opt.k_index, total_iters) if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_iters))
            model.save_networks(f"k{opt.k_index}_latest")
            model.save_networks(f"k{opt.k_index}_{epoch}")

        if epoch % opt.save_epoch_freq == 0:
            save_png = True
        else:
            save_png=False

        val_score = model.get_test_score(
            opt,valid_dataset, visualizer, f"Val - {opt.name}", 222,save_png)
        model.change_lr(val_score)
        test_score = model.get_test_score(
            opt,test_dataset, visualizer, f"Test - {opt.name}", 333,save_png)

        losses = OrderedDict(
            [(f"K:{opt.k_index} Val Score", val_score), (f"K:{opt.k_index} Test Score", test_score)])
        test_scores['X'].append(epoch)
        test_scores['Y'].append([losses[k] for k in test_scores['legend']])

        _X = np.stack([np.array(test_scores['X'])] *
                      len(test_scores['legend']), 1)
        _Y = np.array(test_scores['Y'])

        if opt.display_id > 0:
            visualizer.vis.line(
                X=_X,
                Y=_Y,
                opts={
                    'title': opt.name + ' val/test scores over time',
                    'legend': test_scores['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=111+opt.k_index)

        train_score = np.mean(np.array(train_loss_list))

        with open(f"./scores/{name}.txt", "a") as f:
            f.write(str(opt.k_index)+","+str(epoch)+","+str(train_score)+"," +
                    str(val_score)+","+str(test_score)+"\n")  # 自带文件关闭功能，不需要再写f.close() 记得添加换行符！

        print(
            f"mean MAE:{test_score} HU. Min MAE :{best_score} in {best_epoch}")
        if test_score < best_score:
            best_score = test_score
            best_epoch = epoch
            print(f"Saved Best!")
            model.save_networks(f"k{opt.k_index}_best")
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch,
                                                              opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        print(f"Train score: {train_score}")
        print(f"Val score: {val_score}")
        print(f"Test score: {test_score}")

    return train_score, val_score, test_score


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    torch.cuda.empty_cache()
    # Set fixed random number seed
    setup_seed(123)
    train_loss_sum, valid_loss_sum = 0, 0
    # create a visualizer that display/save images and plots
    val_score_total, test_score_total, train_score_total = 0, 0, 0
    if opt.single_k_index >= 0:
        i = opt.single_k_index
        # create a dataset given opt.dataset_mode and other options
        opt.k_index = i
        visualizer = Visualizer(opt)
        Visualizer.k_index = i
        opt.k_type="train"
        train_dataset = create_dataset(opt)
        opt = TrainOptions().parse()   # get training options
        opt.k_type = "valid"
        opt.k_index = i
        # create a dataset given opt.dataset_mode and other options
        valid_dataset = create_dataset(opt)
        opt_test = TestOptions().parse()
        opt_test.k_index = i
        opt_test.num_threads = opt.num_threads
        opt_test.batch_size = opt.batch_size
        opt_test.serial_batches = True
        opt_test.no_flip = True
        opt_test.display_id = '-1'
        opt_test.phase = "test"
        opt_test.k_type = "test"
        test_dataset = create_dataset(opt_test)

        train_dataset_size = len(train_dataset)
        val_dataset_size = len(valid_dataset)
        test_dataset_size = len(test_dataset)
        print('The number of training images = %d' % train_dataset_size)
        print('The number of val images = %d' % val_dataset_size)
        print('The number of test images = %d' % test_dataset_size)
        # create a model given opt.model and other options
        model = create_model(opt)
        # regular setup: load and print networks; create schedulers
        model.setup(opt)
        train_score, val_score, test_score = traink(
            model, visualizer, train_dataset, valid_dataset, opt)
        print(
            f"average score:\n\tTrain:{train_score}\n\tValid:{val_score}\n\tTest:{test_score}")
    else:

        visualizer = Visualizer(opt)
        for i in range(opt.k_fold):
            print('*'*25, 'K = ', i, '', '*'*25)
            Visualizer.k_index = i
            opt.k_type = "train"
            opt.k_index = i
            # create a dataset given opt.dataset_mode and other options
            train_dataset = create_dataset(opt)
            opt = TrainOptions().parse()   # get training options
            opt.k_type = "valid"
            opt.k_index = i
            # create a dataset given opt.dataset_mode and other options

            valid_dataset = create_dataset(opt)
            opt_test = TestOptions().parse()
            opt_test.k_index = i
            opt_test.num_threads = opt.num_threads
            opt_test.batch_size = opt.batch_size
            opt_test.serial_batches = True
            opt_test.no_flip = True
            opt_test.display_id = '-1'
            opt_test.phase = "test"
            opt_test.k_type = "test"
            test_dataset = create_dataset(opt_test)

            train_dataset_size = len(train_dataset)
            val_dataset_size = len(valid_dataset)
            test_dataset_size = len(test_dataset)
            print('The number of training images = %d' % train_dataset_size)
            print('The number of val images = %d' % val_dataset_size)
            print('The number of test images = %d' % test_dataset_size)
            # create a model given opt.model and other options
            model = create_model(opt)
            # regular setup: load and print networks; create schedulers
            model.setup(opt)
            train_score, val_score, test_score = traink(
                model, visualizer, train_dataset, valid_dataset, opt)
            train_score_total += train_score
            val_score_total += val_score
            test_score_total += test_score

        print(
            f"average score:\n\tTrain:{train_score_total/5.}\n\tValid:{val_score_total/5.}\n\tTest:{test_score_total/5.}")
