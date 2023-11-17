import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import loss as loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list import ImageList, ImageList_twice
import math
import copy
import utils

def calc_pm(iter_num,  max_iter=5000.0):
    high = 1.0
    low = 0.0
    alpha = 10.0
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

class MyDataset(ImageList):
    def __init__(self, cfg, transform):
        self.btw_data = ImageList(cfg, transform=transform)
        self.imgs = self.btw_data.imgs

    def __getitem__(self, index):
        data, target = self.btw_data[index]
        return data, target, index

    def __len__(self):
        return len(self.btw_data)


class data_batch:
    def __init__(self, gt_data, batch_size: int, drop_last: bool, randomize_flag: bool, num_class: int, num_batch: int) -> None:
        # if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
        #         batch_size <= 0:
        #     raise ValueError("batch_size should be a positive integer value, "
        #                      "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        gt_data = gt_data.astype(dtype=int)

        self.class_num = num_class
        self.batch_num = num_batch

        self.norm_mode = False
        self.all_data = np.arange(len(gt_data))

        self.data_len = len(gt_data)

        self.norm_mode_len= math.floor(self.data_len/ self.batch_num)


        self.i_range = len(gt_data)
        self.s_list = []
        if randomize_flag == True:
            self.norm_mode = True
            self.set_length(self.norm_mode_len)
            self.i_range = self.norm_mode_len
        else:
            for c_iter in range(self.class_num):
                cur_data = np.where(gt_data == c_iter)[0]
                self.s_list.append(cur_data)
                cur_length = math.floor((len(cur_data) * self.class_num) / self.batch_num)
                if(cur_length < self.data_len):
                    self.set_length(cur_length)
                    self.i_range = len(cur_data)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.prob_mat = np.zeros(())
        self.idx = 0
        self.c_iter = 0
        self.drop_class = set()

    def shuffle_list(self):
        for c_iter in range(self.class_num):
            np.random.shuffle(self.s_list[c_iter])

    def set_length(self, length: int):
        self.data_len = length

    def set_probmatrix(self, prob_mat):
        self.prob_mat = prob_mat

    def get_list(self):
        found_split = False
        self.norm_mode = False
        winList = np.argmax(self.prob_mat, axis=1)
        break_ctr = 0
        for c_iter in range(self.class_num):
            cur_data = np.where(winList == c_iter)[0]
            # num_gt = np.sum(winList == c_iter)
            # print(str(c_iter) + " : " + str(num_gt))
            self.s_list.append(cur_data)
            cur_length = math.floor((len(cur_data) * self.class_num) / self.batch_num)
            if (cur_length < 1):
                self.drop_class.add(c_iter)
                continue
            if (cur_length < self.data_len):
                self.set_length(cur_length)
                self.i_range = len(cur_data)
        if(len(self.drop_class) > 0):
            cur_length = math.floor((self.i_range * (self.class_num-len(self.drop_class))) / self.batch_num)
            self.set_length(cur_length)
        return True


    def __iter__(self):
        batch = []
        bs = self.batch_num
        if(self.norm_mode):
            while(True):
                np.random.shuffle(self.all_data)
                for idx in range(self.i_range):
                    for b_iter in range(bs):
                        batch.append(self.all_data[idx*bs+b_iter])
                    yield batch
                    batch = []
        else:
            batch_ctr = 0
            cur_ctr = 0
            pick_item = np.arange(self.class_num)
            while(True):
                new_round = False
                for idx in range(self.i_range):
                    if(new_round):
                        break
                    np.random.shuffle(pick_item)
                    for c_iter in range(self.class_num):
                        if(new_round):
                            break
                        c_iter_l = pick_item[c_iter]
                        if c_iter_l in self.drop_class:
                            # print(self.drop_class)
                            continue
                        c_idx = idx % len(self.s_list[c_iter_l])
                        batch.append(self.s_list[c_iter_l][c_idx])
                        cur_ctr += 1
                        if(cur_ctr % bs == 0):
                            yield batch
                            batch = []
                            cur_ctr = 0
                            batch_ctr += 1
                            if(batch_ctr == self.data_len):
                                batch_ctr = 0
                                self.shuffle_list()
                                new_round = True


    def __len__(self):
        return self.data_len

    def get_range(self):
        return self.i_range

def get_L2norm_loss_self_driven(x):
    l = (x.norm(p=2, dim=1).mean() - 25.0) ** 2
    return 0.05 * l

def SM(Xs,Xt,Ys,Yt,Cs_memory,Ct_memory,Wt=None,decay=0.3):
    Cs=Cs_memory.clone()
    Ct=Ct_memory.clone()

    # Cs = torch.nn.functional.normalize(Cs)
    # Ct = torch.nn.functional.normalize(Ct)
    # Xs = torch.nn.functional.normalize(Xs)
    # Xt = torch.nn.functional.normalize(Xt)

    K=Cs.size(0)
    for k in range(K):
        dec_s = True
        dec_t = True
        Xs_k=Xs[Ys==k]
        Xt_k=Xt[Yt==k]
        if len(Xs_k)==0:
            Cs_k=0.0
            dec_s = False
        else:
            Cs_k=torch.mean(Xs_k,dim=0)

        if len(Xt_k)==0:
            Ct_k=0.0
            dec_t = False
        else:
            Ct_k=torch.mean(Xt_k,dim=0)
        if(dec_s):
            Cs[k,:]=(1-decay)*Cs_memory[k,:]+decay*Cs_k
        if(dec_t):
            Ct[k,:]=(1-decay)*Ct_memory[k,:]+decay*Ct_k

    # Cs = torch.nn.functional.normalize(Cs)
    # Ct = torch.nn.functional.normalize(Ct)
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    inter_dist=0

    intra_dist_ss = 0
    intra_dist_st = 0
    intra_dist_tt = 0

    Cs_norm = torch.nn.functional.normalize(Cs)
    Ct_norm = torch.nn.functional.normalize(Ct)

    for k_i in range(K):
        for k_j in range(K):
            if k_i != k_j:
                intra_dist_st += torch.dot(Cs_norm[k_i], Ct_norm[k_j])
                intra_dist_ss += torch.dot(Cs_norm[k_i], Cs_norm[k_j])
                intra_dist_tt += torch.dot(Ct_norm[k_i], Ct_norm[k_j])
            else:
                inter_dist += torch.dot(Cs[k_i]-Ct[k_j], Cs[k_i]-Ct[k_j])
    intra_dist_st = intra_dist_st / ((K - 1) * (K - 1))
    intra_dist_ss = intra_dist_ss / ((K - 1) * (K - 1))
    intra_dist_tt = intra_dist_tt / ((K - 1) * (K - 1))

    inter_dist = inter_dist / K

    return inter_dist + intra_dist_st + intra_dist_tt, intra_dist_ss, Cs, Ct

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                feats = []
                for j in range(10):
                    feat, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                    feats.append(nn.Softmax(dim=1)(feat))
                outputs = sum(outputs)/10.0
                feats = sum(feats)/10.0
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    all_feat_out = feats.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_feat_out = torch.cat((all_feat_out, feats.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _, outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, all_output, all_feat_out, all_label

def image_classification_test_src(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test_src'][i]) for i in range(10)]
            for i in range(len(loader['test_src'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = torch.from_numpy(np.array(data[0][1]).astype(int))
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                feats = []
                for j in range(10):
                    feat, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                    feats.append(nn.Softmax(dim=1)(feat))
                outputs = sum(outputs)/10.0
                feats = sum(feats)/10.0
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    all_feat_out = feats.float().cpu()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_feat_out = torch.cat((all_feat_out, feats.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test_src"])
            for i in range(len(loader['test_src'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _, outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, all_output, all_feat_out, all_label

def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=16, drop_last=True)

    dsets["target"] = ImageList_twice(open(data_config["target"]["list_path"]).readlines(), transform=[prep_dict["target"],prep_dict["target"]])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=16, drop_last=True)

    class_num = config["network"]["params"]["class_num"]

    if(args.run_id != 0 and args.perc > 0.005):
        loadFile = open(config["load_stem"] + str(args.run_id - 1) + ".npy", 'rb')
        tar_pseu_load = np.load(loadFile)

        tar_win_row = np.max(tar_pseu_load, axis=1)
        move_iter = int(round(tar_pseu_load.shape[0] * args.perc))
        tts_ind = np.argpartition(tar_win_row, -move_iter)[-move_iter:]
        tts_classes = np.argmax(tar_pseu_load[tts_ind], axis=1)



        t_source = np.array(dsets["source"].imgs)
        t_target = np.array(dsets["target"].imgs)

        tts_items = t_target[tts_ind]
        tts_items[:,1] = tts_classes

        t_source = np.append(t_source, tts_items, axis = 0)
        #t_target = np.delete(t_target, tts_ind, 0)
        #tar_pseu_load = np.delete(tar_pseu_load, tts_ind, 0)

        s_gt = t_source[:, 1]
        t_gt = t_target[:, 1]

#tuple(map(tuple, arr))
        t_source = list(map(tuple,t_source))
        t_target = list(map(tuple,t_target))

        dsets["source"].imgs = t_source
        dsets["target"].imgs = t_target
    else:
        if(args.run_id != 0):
            loadFile = open(config["load_stem"] + str(args.run_id - 1) + ".npy", 'rb')
            tar_pseu_load = np.load(loadFile)
        s_gt = np.array(dsets["source"].imgs)[:, 1]
        t_gt = np.array(dsets["target"].imgs)[:, 1]

    data_batch_source = data_batch(s_gt, batch_size=train_bs, drop_last=False, randomize_flag=False, num_class=class_num, num_batch=train_bs)
    data_batch_target = data_batch(t_gt, batch_size=train_bs, drop_last=False, randomize_flag=True, num_class=class_num, num_batch=train_bs)

    dataloader_source = torch.utils.data.DataLoader(
        dataset=dsets["source"],
        batch_sampler=data_batch_source,
        shuffle=False,
        num_workers=16,
        drop_last=False)

    dataloader_target = torch.utils.data.DataLoader(
        dataset=dsets["target"],
        batch_sampler=data_batch_target,
        shuffle=False,
        num_workers=16,
        drop_last=False)

    if prep_config["test_10crop"]:
        temp = []
        for j in range(10):
            temp.append(copy.deepcopy(dsets["source"]))
            temp[j].transform = prep_dict["test"][j]
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=16) for dset in dsets['test']]

            dsets["test_src"] = [temp[i] for i in range(10)]
            dset_loaders["test_src"] = [DataLoader(dset, batch_size=test_bs, \
                                                   shuffle=False, num_workers=16) for dset in dsets['test_src']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=16)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        if config['method'] == 'DANN':
            ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
        elif config['method'] == 'BIWA':
            ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
        else:
            ad_net = network.AdversarialNetwork(base_network.output_num() * (class_num), 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    #if(args.run_id != 0):
    #    data_batch_target.set_probmatrix(tar_pseu_load)
    #    data_batch_target.get_list()

    ## train   
    len_train_source = len(dataloader_source)
    len_train_target = len(dataloader_target)

    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    targ_iter = 0

    Cs_memory = torch.zeros(class_num, 256).cuda()
    Ct_memory = torch.zeros(class_num, 256).cuda()
    #if (args.run_id != 0):
    #    tar_pseu_prev = torch.from_numpy(tar_pseu_load).clone()
    #    tar_pseu_prev = tar_pseu_prev.cuda()

    iter_source = iter(dataloader_source)
    iter_target = iter(dataloader_target)

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc, tar_prec_vec, tar_feat_vec, tar_lab_vec = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])

            _, src_prec_vec, src_feat_vec, src_lab_vec = image_classification_test_src(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])

            src_lab_vec = src_lab_vec.int()
            tar_lab_vec = tar_lab_vec.int()

            # print(src_feat_vec.shape)

            initial_centers_array = torch.zeros(class_num, 256).cuda()
            for cid in range(class_num):
                # print("TEST")
                # print(src_feat_vec[src_lab_vec == cid])
                # print(src_feat_vec[src_lab_vec == cid].shape)
                # print(torch.sum(src_feat_vec[src_lab_vec == cid, :], dim=0))
                # print(torch.sum(src_lab_vec == cid).float())
                # print("now ok?")
                initial_centers_array[cid] = torch.sum(src_feat_vec[src_lab_vec == cid], dim = 0) / torch.sum(src_lab_vec == cid).float()

            initial_centers_array = initial_centers_array.cpu()
            soft_label_lp, hard_label_lp, acc_lp = utils.get_labels_from_lp(src_feat_vec, utils.to_onehot(src_lab_vec, class_num), tar_feat_vec, tar_lab_vec, class_num)
            soft_label_kmeans, hard_label_kmeans, acc_kmeans = utils.get_labels_from_kmeans(initial_centers_array, tar_feat_vec, class_num, tar_lab_vec)
            soft_label_nn, hard_label_nn, acc_nn = utils.get_labels_from_nn(tar_feat_vec, tar_prec_vec, class_num, tar_lab_vec)

            # print(tar_prec_vec.shape)
            # print(soft_label_lp.shape)
            # print(soft_label_kmeans.shape)
            # print(soft_label_nn.shape)
            #
            # print(torch.sum(tar_prec_vec, dim=1))
            # print(torch.sum(soft_label_lp, dim=1))
            # print(torch.sum(soft_label_kmeans, dim=1))
            # print(torch.sum(soft_label_nn, dim=1))

            soft_label_conf = tar_prec_vec

            soft_3 = (soft_label_conf + soft_label_lp + soft_label_kmeans) / 3
            soft_4 = (soft_label_conf + soft_label_lp + soft_label_kmeans + soft_label_nn) / 4

            _, pred_3 = torch.max(soft_3, 1)
            acc_3 = torch.sum(torch.squeeze(pred_3).float() == tar_lab_vec).item() / float(tar_lab_vec.size()[0]) * 100
            _, pred_4 = torch.max(soft_4, 1)
            acc_4 = torch.sum(torch.squeeze(pred_4).float() == tar_lab_vec).item() / float(tar_lab_vec.size()[0]) * 100

            log_str = "iter: {:05d}, p_conf: {:.5f} p_lp: {:.5f} p_km: {:.5f} p_nn: {:.5f} p_3: {:.5f} p_4: {:.5f}".format(i, temp_acc*100, acc_lp.item(), acc_kmeans.item(), acc_nn.item(), acc_3, acc_4)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)

            if(args.pl_ver == 0):
                tar_prec_vec = soft_label_conf
            elif(args.pl_ver == 1):
                tar_prec_vec = (soft_label_conf + soft_label_lp + soft_label_nn) / 3
            # elif (args.pl_ver == 2):
            #     tar_prec_vec = soft_label_kmeans
            # elif (args.pl_ver == 3):
            #     tar_prec_vec = soft_label_nn
            # elif (args.pl_ver == 4):
            #     tar_prec_vec = (soft_label_conf + soft_label_lp + soft_label_kmeans + soft_label_nn) / 4
            # elif (args.pl_ver == 5):
            #     tar_prec_vec = (soft_label_conf + soft_label_lp + soft_label_kmeans) / 3
            # elif (args.pl_ver == 6):
            #     tar_prec_vec = (soft_label_conf + soft_label_nn) / 2
            # elif (args.pl_ver == 7):
            #     tar_prec_vec = soft_label_lp
            # elif (args.pl_ver == 8):
            #     tar_prec_vec = (soft_label_conf + soft_label_lp) / 2

            tar_prec_vec = tar_prec_vec.data.cpu().numpy()
            temp_model = nn.Sequential(base_network)
            saveFile = open(config["save_labels"], 'wb')
            np.save(saveFile, tar_prec_vec)
            saveFile.close()
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], str(args.run_id) + "_" + str(i)+ "_model.pth.tar"))

        loss_params = config["loss"]                  

        targ_iter+=1

        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        inputs_source, labels_source = iter_source.next()
        inputs_target_c, _, sample_idx = iter_target.next()
        inputs_target = inputs_target_c[0]
        # labels_target = labels_target_c[0]
        inputs_target_2 = inputs_target_c[1]
        labels_source  = torch.from_numpy(np.array(labels_source).astype(int))
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        inputs_target_2 = inputs_target_2.cuda()
        ### CHANGE ###
        base_network.del_gradient()
        base_network.zero_grad()
        features_source, outputs_source = base_network.bp(inputs_source)
        features_target, outputs_target = base_network.bp(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)

        outputs_target_sm = nn.Softmax(dim=1)(outputs_target)

        c_, pseu_labels_target = torch.max(outputs_target_sm, 1)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        batch_size = outputs_target.shape[0]
        mask_target = torch.ones(batch_size, base_network.output_num())
        mask_target = mask_target.cuda()
        mask_target = mask_target.detach()
        batch_size = outputs_source.shape[0]
        mask_source = torch.ones(batch_size, base_network.output_num())
        mask_source = mask_source.cuda()
        mask_source = mask_source.detach()
        mask = torch.cat((mask_source, mask_target), dim=0)

        mask_source = torch.mul(features_source, mask_source)
        mask_target = torch.mul(features_target, mask_target)
        lam = network.calc_coeff(i)
        softmax_out = nn.Softmax(dim=1)(outputs)
        if config['method'] == 'CDAN+E':           
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
            lam = network.calc_coeff(i)
            loss_sm, loss_sm_ss, Cs_memory, Ct_memory = SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory)
            transfer_loss = transfer_loss + args.alpha * lam * loss_sm + loss_sm_ss
        elif config['method']  == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
            loss_sm, loss_sm_ss, Cs_memory, Ct_memory = SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory)
            transfer_loss = transfer_loss + args.alpha * lam * loss_sm + loss_sm_ss
        elif config['method']  == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
            lam = network.calc_coeff(i)
            loss_sm, loss_sm_ss, Cs_memory, Ct_memory = SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory)
            transfer_loss = transfer_loss + args.alpha * lam * loss_sm + loss_sm_ss #+ lam * s_fc2_L2norm_loss + lam * t_fc2_L2norm_loss
        elif config['method']  == 'BIWA':
            transfer_loss = loss.DANN(mask, ad_net)
            lam = network.calc_coeff(i)
            loss_sm, loss_sm_ss, Cs_memory, Ct_memory = SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory)
            transfer_loss = transfer_loss + args.alpha * lam * loss_sm + loss_sm_ss #+ lam * s_fc2_L2norm_loss + lam * t_fc2_L2norm_loss
        else:
            raise ValueError('Method cannot be recognized.')
        #classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        targets_s = torch.zeros(batch_size, class_num).cuda().scatter_(1, labels_source.view(-1, 1), 1)
        # targets_s = targets_s.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            feat_u2, outputs_u2 = base_network(inputs_target_2)
            p = (torch.softmax(outputs_target, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/0.5)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        all_inputs = torch.cat([inputs_source, inputs_target, inputs_target_2], dim=0)

        all_targets = torch.cat([targets_s, targets_u, targets_u], dim=0)
        l = np.random.beta(0.75, 0.75)
        l = max(l, 1-l)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = utils.interleave(mixed_input, batch_size)
        # s = [sa, sb, sc]
        # t1 = [t1a, t1b, t1c]
        # t2 = [t2a, t2b, t2c]
        # => s' = [sa, t1b, t2c]   t1' = [t1a, sb, t1c]   t2' = [t2a, t2b, sc]

        feat_mu, logits = base_network(mixed_input[0])
        logits = [logits]
        for input in mixed_input[1:]:
            _, temp = base_network(input)
            logits.append(temp)

        # put interleaved samples back
        # [i[:,0] for i in aa]
        logits = utils.interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        train_criterion = utils.SemiLoss()

        Lx, Lu, w = train_criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
            i, config["num_iterations"], 100)

        loss_mm = lam * (Lx + w * Lu)

        if(args.mm_ver == 1):
            transfer_loss += loss_mm

        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()

        Cs_memory.detach_()
        Ct_memory.detach_()
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN', 'BIWA'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13", "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'domain-net', 'visda', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../../data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../../data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--num_iter', type=int, default=5004, help="interval of two continuous test phase")
    parser.add_argument('--test_interval', type=int, default=5000, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--perc', type=float, default=0.1, help="Target to source percentage")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--run_id', type=int, default=0, help="Norm factor")
    parser.add_argument('--alpha', type=float, default=1.0, help="Norm factor")

    parser.add_argument('--mm_ver', type=int, default=1, help="MixMatch loss active")
    parser.add_argument('--pl_ver', type=int, default=1, help="MixMatch loss active")
    parser.add_argument('--mb_ver', type=int, default=1, help="MixMatch loss active")

    args = parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    # Set random number seed.
    np.random.seed(args.seed + 100 * args.run_id)
    torch.manual_seed(args.seed + 100 * args.run_id)

    if(args.mb_ver == 0):
        import network as network
    else:
        import network_mb as network

    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iter
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])

    config["out_file"] = open(osp.join(config["output_path"], "log-" + str(args.run_id) + ".txt"), "w")
    config["save_labels"] = osp.join(config["output_path"], "logits-" + str(args.run_id) + ".npy")
    config["load_stem"] = osp.join(config["output_path"], "logits-")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                      "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                      "test":{"list_path":args.t_dset_path, "batch_size":36}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
           ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
           ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
             ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters       
        config["network"]["params"]["class_num"] = 31 
    elif config["dataset"] == "domain-net":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 40
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config)+"\n")
    config["out_file"].flush()
    train(config)
