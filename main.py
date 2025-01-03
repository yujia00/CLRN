import numpy as np
from numpy import random
import random
import datetime
import torch as t
import os
import torch.nn as nn
import torch.utils.data as dataloader
import Timelogger
import AutoGNN
import graph_utils
import DataHandler
import pickle
from Params import args
from tqdm import tqdm

class Model():
    def __init__(self):
        self.trn_file = f"{args.path}{args.dataset}/trn_"
        self.tst_file = f"{args.path}{args.dataset}/tst_int"
        self.t_max = -1
        self.t_min = 0x7FFFFFFF
        self.time_number = -1
        self.user_num = -1
        self.item_num = -1

        self.behavior_mats = {}
        self.behaviors = []
        self.behaviors_data = {}
        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.curEpoch = 0
        self.beta_lr = 1

        self._initialize_behaviors()
        self._load_behavior_data()


        print("Start building behavior matrices:", datetime.datetime.now())
        self._build_behavior_matrices()
        print("End building behavior matrices:", datetime.datetime.now())

        print(f"user_num: {self.user_num}")
        print(f"item_num: {self.item_num}\n")


        self._build_data_loaders()

    def _initialize_behaviors(self):

        if args.dataset == 'Tmall':
            self.behaviors_SSL = ['pv', 'fav', 'cart', 'buy']
            self.behaviors = ['pv', 'fav', 'cart', 'buy']
        elif args.dataset == 'IJCAI_15':
            self.behaviors = ['click', 'fav', 'cart', 'buy']
            self.behaviors_SSL = ['click', 'fav', 'cart', 'buy']
        elif args.dataset == 'retailrocket':
            self.behaviors = ['view', 'cart', 'buy']
            self.behaviors_SSL = ['view', 'cart', 'buy']

    def _load_behavior_data(self):

        for i, behavior in enumerate(self.behaviors):
            with open(f"{self.trn_file}{behavior}", 'rb') as fs:
                data = pickle.load(fs)
                self.behaviors_data[i] = data


                self.user_num = max(self.user_num, data.get_shape()[0])
                self.item_num = max(self.item_num, data.get_shape()[1])


                self.t_max = max(self.t_max, data.data.max())
                self.t_min = min(self.t_min, data.data.min())

                if behavior == args.target:
                    self.trainMat = data
                    self.trainLabel = 1 * (self.trainMat != 0)
                    buy_u, buy_v = self.trainMat.nonzero()
                    self.buy_array = np.stack((buy_u, buy_v), axis=1)
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))

    def _build_behavior_matrices(self):
        for i in range(len(self.behaviors)):
            self.behavior_mats[i] = graph_utils.get_use(self.behaviors_data[i])

    def _build_data_loaders(self):
        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1, 1))).tolist()
        train_dataset = DataHandler.RecDataset_beh(
            self.behaviors, train_data, self.item_num, self.behaviors_data, True
        )
        self.train_loader = dataloader.DataLoader(
            train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True
        )
        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)
        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        test_data = np.hstack((test_user.reshape(-1, 1), test_item.reshape(-1, 1))).tolist()
        test_dataset = DataHandler.RecDataset(
            test_data, self.item_num, self.trainMat, 0, False
        )
        self.test_loader = dataloader.DataLoader(
            test_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True
        )

    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u, i), dim=1) * args.inner_product_mult
        pred_j = t.sum(t.mul(u, j), dim=1) * args.inner_product_mult
        return pred_i, pred_j

    def SSL(self, user_embeddings, sampled_user_indices):

        def compute_negative_sample_scores(batch_indices, all_indices, embedding1, embedding2):
            all_index_set = set(np.array(all_indices.cpu()))
            batch_index_set = set(np.array(batch_indices.cpu()))
            negative_index_set = all_index_set - batch_index_set
            negative_indices = t.as_tensor(np.array(list(negative_index_set))).long().cuda()

            negative_indices = t.unsqueeze(negative_indices, 0).repeat(len(batch_indices), 1)
            negative_indices = t.reshape(negative_indices, (1, -1)).squeeze()

            positive_indices = batch_indices.long().cuda()
            positive_indices = t.unsqueeze(positive_indices, 1).repeat(1, len(negative_index_set))
            positive_indices = t.reshape(positive_indices, (1, -1)).squeeze()

            negative_scores = t.sum(
                compute_pairwise_scores(embedding1, embedding2, positive_indices, negative_indices)
                .squeeze()
                .view(len(batch_indices), -1),
                -1,
            )
            return negative_scores

        def compute_pairwise_scores(embedding1, embedding2, positive_indices=None, negative_indices=None):
            if positive_indices is not None:
                embedding1 = embedding1[positive_indices]
                embedding2 = embedding2[negative_indices]

            batch_size = embedding1.shape[0]
            embedding_dim = embedding1.shape[1]
            scores = t.exp(
                t.div(
                    t.bmm(embedding1.view(batch_size, 1, embedding_dim),
                          embedding2.view(batch_size, embedding_dim, 1)).view(batch_size, 1),
                    np.power(embedding_dim, 1) + 1e-8,
                )
            )
            return scores

        def compute_infoNCE_loss(embedding1, embedding2, all_indices):
            num_samples = all_indices.shape[0]
            positive_scores = compute_pairwise_scores(embedding1[all_indices], embedding2[all_indices]).squeeze()
            negative_scores = t.zeros((num_samples,), dtype=t.float64).cuda()

            num_steps = int(np.ceil(num_samples / args.SSL_batch))
            for step in range(num_steps):
                start_idx = step * args.SSL_batch
                end_idx = min((step + 1) * args.SSL_batch, num_samples)
                batch_indices = all_indices[start_idx:end_idx]
                batch_negative_scores = compute_negative_sample_scores(batch_indices, all_indices, embedding1,
                                                                       embedding2)

                if step == 0:
                    negative_scores = batch_negative_scores
                else:
                    negative_scores = t.cat((negative_scores, batch_negative_scores), 0)

            contrastive_loss = -t.log(1e-8 + t.div(positive_scores, negative_scores + 1e-8))
            assert not t.any(t.isnan(contrastive_loss))
            assert not t.any(t.isinf(contrastive_loss))
            return t.where(t.isnan(contrastive_loss), t.full_like(contrastive_loss, 1e-8), contrastive_loss)

        user_contrastive_loss_list = []
        sampled_length = int(sampled_user_indices.shape[0] / 10)
        sampled_user_indices = t.as_tensor(
            np.random.choice(sampled_user_indices.cpu(), size=sampled_length, replace=False, p=None)
        ).cuda()

        for behavior_index in range(len(self.behaviors_SSL)):
            user_contrastive_loss_list.append(
                compute_infoNCE_loss(user_embeddings[-1], user_embeddings[behavior_index], sampled_user_indices)
            )
        return user_contrastive_loss_list, sampled_user_indices

    def run(self):
        if args.isload:
            print("---------------------- Pre-test:")
            HR, NDCG = self.testEpoch(self.test_loader)
            print(f"HR: {HR}, NDCG: {NDCG}")
        log('Model Prepared')

        cvWait = 0
        self.best_HR = 0
        self.best_NDCG = 0
        self.user_embed, self.item_embed = None, None
        self.user_embeds, self.item_embeds = None, None

        print("Test before train:")
        HR, NDCG = self.testEpoch(self.test_loader)


        for epoch in range(self.curEpoch, args.epoch + 1):
            self.curEpoch = epoch
            log(f"***************** Start epoch: {epoch} ************************")

            if not args.isJustTest:
                epoch_loss, user_embed, item_embed, user_embeds, item_embeds = self.trainEpoch(args.alpha)
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            self.scheduler.step()


            is_best_HR = self._update_best_metrics(HR, "HR", user_embed, item_embed, user_embeds, item_embeds)
            is_best_NDCG = self._update_best_metrics(NDCG, "NDCG", user_embed, item_embed, user_embeds, item_embeds)


            if not is_best_HR and not is_best_NDCG:
                cvWait += 1

            if cvWait == args.patience:
                print(f"Early stop at epoch {self.best_epoch}: best HR: {self.best_HR}, best NDCG: {self.best_NDCG}\n")
                self.saveHistory()
                break

        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)

    def _update_best_metrics(self, value, metric_name, user_embed, item_embed, user_embeds, item_embeds):
        if metric_name == "HR" and value > self.best_HR:
            self.best_HR = value
            self.best_epoch = self.curEpoch
            self._save_best_embeddings(user_embed, item_embed, user_embeds, item_embeds)
            print(
                f"-------------------------------------------------------------------------------------------------------------------------- Best HR: {self.best_HR}")
            self.saveHistory()
            return True

        if metric_name == "NDCG" and value > self.best_NDCG:
            self.best_NDCG = value
            self.best_epoch = self.curEpoch
            self._save_best_embeddings(user_embed, item_embed, user_embeds, item_embeds)
            print(
                f"-------------------------------------------------------------------------------------------------------------------------- Best NDCG: {self.best_NDCG}")
            self.saveHistory()
            return True
        return False

    def _save_best_embeddings(self, user_embed, item_embed, user_embeds, item_embeds):
        self.user_embed = user_embed
        self.item_embed = item_embed
        self.user_embeds = user_embeds
        self.item_embeds = item_embeds

    def saveHistory(self):
        history = dict()
        history['loss'] = self.train_loss
        history['HR'] = self.his_hr
        history['NDCG'] = self.his_ndcg
        ModelName = self.modelName
        with open(r'./History/' + args.dataset + r'/' + ModelName + '.his', 'wb') as fs:
            pickle.dump(history, fs)


    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[
            batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = []
        item_id_pos = []
        item_id_neg = []
        cur = 0

        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])
            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item())
                item_id_neg.append(neglocs[j])
                cur += 1
        return t.as_tensor(np.array(user_id)).cuda(), t.as_tensor(np.array(item_id_pos)).cuda(), t.as_tensor(
            np.array(item_id_neg)).cuda()

    # Hard negative sample
    def buy_ng_sample(self, batch_users, buy_user_embed, buy_item_embed):
        batch = np.array(batch_users.cpu(), dtype=np.int32)
        batch_array = self.buy_array[batch]
        length = len(batch_array)
        neg_buy_items = []
        for user, _ in batch_array:
            negitems =[]
            for i in range(args.mum):
                while True:
                    negitem = random.choice(range(self.item_num))
                    if negitem not in self.trainMat[user].nonzero()[1]:
                        break
                negitems.append(negitem)
            neg_buy_items.append(negitems)
        neg_buy_items = t.tensor(neg_buy_items)
        buy_user_emb = buy_user_embed[batch_array[:, 0]].clone().detach()
        buy_pos_emb = buy_item_embed[batch_array[:, 1]].clone().detach()
        buy_neg_emb = buy_item_embed[neg_buy_items]

        seed = t.rand(length, 1, buy_pos_emb.shape[1], 1).to(buy_pos_emb.device)
        buy_neg_emb_ = seed * buy_pos_emb.unsqueeze(1) + (1 - seed) * buy_neg_emb
        scores = (buy_user_emb.unsqueeze(dim=1) * buy_neg_emb_).sum(dim=-1)

        indices = t.max(scores, dim=1)[1].detach()
        neg_items_emb_ = buy_neg_emb_.permute([0, 2, 1, 3])
        return buy_user_emb, buy_pos_emb, neg_items_emb_[[[i] for i in range(length)],
                                          range(neg_items_emb_.shape[1]), indices,
                                          :]

    def trainEpoch(self, alpha):
        train_loader = self.train_loader
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample_multi()
        time = datetime.datetime.now()
        print("end_ng_samp:  ", time)
        epoch_loss = 0


        self.behavior_loss_list = [None] * len(self.behaviors)
        self.user_id_list = [None] * len(self.behaviors)
        self.item_id_pos_list = [None] * len(self.behaviors)
        self.item_id_neg_list = [None] * len(self.behaviors)

        cnt = 0
        for idx, (user, item_i, item_j) in enumerate(tqdm(train_loader)):
            user = user.long().cuda()
            self.user_step_index = user

            user_embed, item_embed, user_embeds, item_embeds, buy3_user_embed, buy3_item_embed = self.model()

            for index in range(len(self.behaviors)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]
                self.user_id_list[index] = user[not_zero_index].long().cuda()
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()
                userEmbed = user_embed[self.user_id_list[index]]
                posEmbed = item_embed[self.item_id_pos_list[index]]
                negEmbed = item_embed[self.item_id_neg_list[index]]

                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)
                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()

            infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds,self.user_step_index)

            for i in range(len(self.behaviors)):
                infoNCELoss_list[i] = (infoNCELoss_list[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i]).sum()


            buy_user_emb, buy_pos_emb, neg_items_emb_ = self.buy_ng_sample(self.user_id_list[-1], buy3_user_embed, buy3_item_embed)
            buy_userEmbed = buy_user_emb.mean(dim=1)
            buy_posEmbed = buy_pos_emb.mean(dim=1)
            buy_negEmbed = neg_items_emb_.mean(dim=1)
            buy_pred_i, buy_pred_j = self.innerProduct(buy_userEmbed, buy_posEmbed, buy_negEmbed)

            smooth_factor = 1e-8
            diff = buy_pred_i.view(-1) - buy_pred_j.view(-1)
            smooth_diff = diff.sigmoid() + smooth_factor
            loss = -smooth_diff.log()
            buy_loss= t.sum(loss)

            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = (buy_loss * alpha + bprloss + args.reg * regLoss + args.beta * infoNCELoss) / args.batch
            epoch_loss = epoch_loss + loss.item()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()
            t.cuda.empty_cache()

            cnt += 1
        return epoch_loss, user_embed, item_embed, user_embeds, item_embeds


    def testEpoch(self, data_loader, save=False):
        epochHR, epochNDCG = [0] * 2
        with t.no_grad():
            user_embed, item_embed, user_embeds, item_embeds, buy_user_embed, buy_item_embed = self.model()

        cnt = 0
        tot = 0

        for user, item_i in data_loader:

            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)
            userEmbed = user_embed[user_compute]
            itemEmbed = item_embed[item_compute]
            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)
            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)
            epochHR = epochHR + hit
            epochNDCG = epochNDCG + ndcg

            cnt += 1
            tot += user.shape[0]
        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")
        return result_HR, result_NDCG

    def calcRes(self, pred_i, user_item1,user_item100):
        hit = 0
        ndcg = 0
        for j in range(pred_i.shape[0]):
            _, shoot_index = t.topk(pred_i[j],
                                    args.shoot)
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()
            if type(shoot) != int and (user_item1[j] in shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(user_item1[j]) + 2))
            elif type(shoot) == int and (user_item1[j] == shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(0 + 2))
        return hit, ndcg

    def sampleTestBatch(self, batch_user_id, batch_item_id):

        batch = len(batch_user_id)
        tmplen = (batch * 100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()
        user_item1 = batch_item_id
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i]
            negset = np.reshape(np.argwhere(sub_trainMat[i] == 0), [-1])

            random_neg_sam = np.random.permutation(negset)[:99]
            user_item100_one_user = np.concatenate((random_neg_sam, np.array([pos_item])))
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1
        return user_compute, item_compute, user_item1, user_item100

    def setRandomSeed(self):
        np.random.seed(args.seed)
        t.manual_seed(args.seed)
        t.cuda.manual_seed(args.seed)
        random.seed(args.seed)


    def getModelName(self):
        ModelName = \
            "CLRN-" + args.dataset + \
            "-lr-" + str(args.lr) + \
            "-gnn_layer-" + str(args.gnn_layer) + \
            "-hidden_dim-" + str(args.hidden_dim)
        return ModelName


    def loadModel(self, loadPath):
        ModelName = self.modelName
        loadPath = loadPath
        checkpoint = t.load(loadPath)
        self.model = checkpoint['model']
        self.curEpoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        self.train_loss = history['loss']
        self.his_hr = history['HR']
        self.his_ndcg = history['NDCG']


if __name__ == '__main__':

    t.autograd.set_detect_anomaly(True)
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    my_model = Model()
    my_model.run()
