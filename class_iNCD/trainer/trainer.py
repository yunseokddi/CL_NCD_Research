import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from utils.utils import AverageMeter, PairEnum, cluster_acc
from utils import ramps
from tensorboard_logger import log_value
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score


class FirstTrainer(object):
    def __init__(self, args, model, model_without_ddp, old_model, labeled_train_loader, mix_train_loader,
                 labeled_eval_loader, unlabeled_eval_loader, all_eval_loader, optimizer,
                 scheduler,
                 criterion_ce, criterion_bce):
        self.args = args
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.old_model = old_model
        self.labeled_train_loader = labeled_train_loader
        self.train_loader = mix_train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion_ce = criterion_ce
        self.criterion_bce = criterion_bce
        self.labeled_eval_loader = labeled_eval_loader
        self.unlabeled_eval_loader = unlabeled_eval_loader
        self.all_eval_loader = all_eval_loader

    def train(self):
        if self.args.labeled_center > 0:
            self.class_mean, self.class_sig, self.class_cov = self.generate_center()
        else:
            self.class_mean, self.class_sig, self.class_cov = None, None, None

        for epoch in range(self.args.epochs):
            self._train_epoch(epoch)

            print("=" * 150)
            print("\t\t\t\tFirst step test")
            print("=" * 150)

            acc_list = []

            print('Head2: test on unlabeled classes')
            self.args.head = 'head2'
            _, ind = self._val_epoch(epoch, self.unlabeled_eval_loader, return_ind=True)

            print('Evaluating on Head1')
            self.args.head = 'head1'

            print('test on labeled classes (test split)')
            acc = self._val_epoch(epoch, self.labeled_eval_loader, cluster=False)
            acc_list.append(acc)

            print('test on unlabeled NEW-1 (test split)')
            acc = self._val_epoch(epoch, self.unlabeled_eval_loader, cluster=False, ind=ind)
            acc_list.append(acc)

            print('test on unlabeled NEW1 (test split) w/ clustering')
            acc = self._val_epoch(epoch, self.unlabeled_eval_loader, cluster=True)
            acc_list.append(acc)

            print('test on all classes w/o clustering (test split)')
            acc = self._val_epoch(epoch, self.all_eval_loader, cluster=False, ind=ind)
            acc_list.append(acc)

            print('test on all classes w/ clustering (test split)')
            acc = self._val_epoch(epoch, self.all_eval_loader, cluster=True)
            acc_list.append(acc)

            print('Evaluating on Head2')
            self.args.head = 'head2'

            print('test on unlabeled classes (train split)')
            acc = self._val_epoch(epoch, self.unlabeled_eval_loader)
            acc_list.append(acc)

            print('test on unlabeled classes (test split)')
            acc = self._val_epoch(epoch, self.unlabeled_eval_loader)
            acc_list.append(acc)

            # print(
            #     'Acc List: Head1->Old, New-1_wo_cluster, New-1_w_cluster, All_wo_cluster, All_w_cluster, Head2->Train, Test')
            # print(acc_list)

            if self.args.tensorboard:
                log_value('Head1->Old ACC', acc_list[0], epoch)
                log_value('New-1_wo_cluster ACC', acc_list[1], epoch)
                log_value('New-1_w_cluster ACC', acc_list[2], epoch)
                log_value('All_wo_cluster ACC', acc_list[3], epoch)
                log_value('All_w_cluster ACC', acc_list[4], epoch)
                log_value('Head2->Train ACC', acc_list[5], epoch)
                log_value('Test ACC', acc_list[6], epoch)

        torch.save(self.model.state_dict(), self.args.model_dir)
        print("model saved to {}.".format(self.args.model_dir))

    def _train_epoch(self, epoch):
        # create loss statistics recorder for each loss
        loss_record = AverageMeter()  # Total loss recorder
        loss_ce_add_record = AverageMeter()  # CE loss recorder
        loss_bce_record = AverageMeter()  # BCE loss recorder
        consistency_loss_record = AverageMeter()  # MSE consistency loss recorder
        loss_kd_record = AverageMeter()  # KD loss recorder

        tq_train = tqdm(self.train_loader, total=len(self.train_loader))

        self.model.train(True)
        self.scheduler.step()
        w = self.args.rampup_coefficient * ramps.sigmoid_rampup(epoch, self.args.rampup_length)

        for (x, x_bar), label, idx in tq_train:
            x = x.to(self.args.device, non_blocking=True)
            x_bar = x_bar.to(self.args.device, non_blocking=True)
            label = label.to(self.args.device, non_blocking=True)

            mask_lb = label < self.args.num_labeled_classes

            # filter out the labeled entries for x, x_bar, label
            # Use only unlabeled dataset
            x = x[~mask_lb]
            x_bar = x_bar[~mask_lb]
            label = label[~mask_lb]

            # normalize the prototypes
            if self.args.l2_classifier:
                self.model.l2_classifier = True
                with torch.no_grad():
                    w_head = self.model.head1.weight.data.clone()
                    w_head = F.normalize(w_head, dim=1, p=2)
                    self.model.head1.weight.copy_(w_head)
                    # if epoch == 5 and w_head_fix is None:
                    #     w_head_fix = w_head[:args.num_labeled_classes, :]
            else:
                self.model.l2_classifier = False

            # output_1 : labeled classifier result
            # output_2 : unlabelled classifier result
            # feat : feature
            output_1, output_2, feat = self.model(x)
            output_1_bar, output_2_bar, feat_bar = self.model(x_bar)

            # use softmax to get the probability distribution for each head
            prob_1, prob_1_bar = F.softmax(output_1, dim=1), F.softmax(output_1_bar, dim=1)
            prob_2, prob_2_bar = F.softmax(output_2, dim=1), F.softmax(output_2_bar, dim=1)

            # calculate rank statistics
            rank_feat = (feat).detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)

            rank_idx_1, rank_idx_2 = PairEnum(rank_idx)
            rank_idx_1, rank_idx_2 = rank_idx_1[:, :self.args.topk], rank_idx_2[:, :self.args.topk]
            rank_idx_1, _ = torch.sort(rank_idx_1, dim=1)
            rank_idx_2, _ = torch.sort(rank_idx_2, dim=1)

            rank_diff = rank_idx_1 - rank_idx_2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)

            target_ulb = torch.ones_like(rank_diff).float().to(self.args.device)
            target_ulb[rank_diff > 0] = -1

            # get the probability distribution of the prediction for head-2
            prob_1_ulb, _ = PairEnum(prob_2)
            _, prob_2_ulb = PairEnum(prob_2_bar)

            # get the pseudo label from head-2
            label = (output_2).detach().max(1)[1] + self.args.num_labeled_classes

            loss_ce_add = w * self.criterion_ce(output_1,
                                                label) / self.args.rampup_coefficient * self.args.increment_coefficient
            loss_bce = self.criterion_bce(prob_1_ulb, prob_2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob_2, prob_2_bar)  # + F.mse_loss(prob1, prob1_bar)

            # record the losses
            loss_ce_add_record.update(loss_ce_add.item(), output_1.size(0))
            loss_bce_record.update(loss_bce.item(), prob_1_ulb.size(0))
            consistency_loss_record.update(consistency_loss.item(), prob_2.size(0))

            if self.args.labeled_center > 0:
                labeled_feats, labeled_labels = self.sample_labeled_features(self.class_mean, self.class_sig)

                if self.args.distributed:
                    labeled_output1 = self.model.module.forward_feat(labeled_feats)

                else:
                    labeled_output1 = self.model.forward_feat(labeled_feats)

                loss_ce_la = self.args.lambda_proto * self.criterion_ce(labeled_output1, labeled_labels)

            else:
                loss_ce_la = 0

            # w_kd is Lambda
            if self.args.w_kd > 0:
                _, _, old_feat = self.old_model(x)
                size_1, size_2 = old_feat.size()

                # eq.(11)
                loss_kd = torch.dist(F.normalize(old_feat.view(size_1 * size_2, 1), dim=0),
                                     F.normalize(feat.view(size_1 * size_2, 1), dim=0)) * self.args.w_kd

            else:
                loss_kd = torch.tensor(0.0)

            # record losses
            loss_kd_record.update(loss_kd.item(), x.size(0))

            loss = loss_bce + loss_ce_add + w * consistency_loss + loss_ce_la + loss_kd

            if self.args.labeled_center > 0 and isinstance(loss_ce_la, torch.Tensor):
                loss_record.update(loss_ce_la.item(), x.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            errors = {
                'Epoch': epoch,
                'Total loss': loss_record.avg,
                'CE loss': loss_ce_add_record.avg,
                'BCE loss': loss_bce_record.avg,
                'MSE loss': consistency_loss_record.avg,
                'KD loss': loss_kd_record.avg
            }

            tq_train.set_postfix(errors)

        if self.args.tensorboard:
            log_value('Train_loss', loss_record.avg, epoch)
            log_value('CE loss', loss_ce_add_record.avg, epoch)
            log_value('BCE loss', loss_bce_record.avg, epoch)
            log_value('MSE loss', consistency_loss_record.avg, epoch)
            log_value('KD loss', loss_kd_record.avg, epoch)

    def _val_epoch(self, epoch, test_loader, cluster=True, ind=None, return_ind=False):
        self.model.eval()

        preds = np.array([])
        targets = np.array([])

        tq_test = tqdm(test_loader, total=len(test_loader))

        for x, label, _ in tq_test:
            x = x.to(self.args.device, non_blocking=True)
            label = label.to(self.args.device, non_blocking=True)

            if self.args.step == 'first' or self.args.test_new == 'new1':
                output_1, output_2, _ = self.model(x)

                # head 1 : labeled dataset
                if self.args.head == 'head1':
                    output = output_1

                else:
                    output = output_2

            else:
                output_1, output_2, output_3, _ = self.model(x, output='test')

                if self.args.head == 'head1':
                    output = output_1

                elif self.args.head == 'head2':
                    output = output_2

                elif self.args.head == 'head3':
                    output = output_3

                else:
                    assert 'Check args head'
                    output = None

            _, pred = output.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())

        if cluster:
            if return_ind:
                acc, ind = cluster_acc(targets.astype(int), preds.astype(int), return_ind)

            else:
                acc = cluster_acc(targets.astype(int), preds.astype(int), return_ind)

            nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)

            print('Epoch {}, Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(epoch, acc, nmi, ari))

        else:
            if ind is not None:
                if self.args.step == 'first':
                    ind = ind[:self.args.num_unlabeled_classes1, :]
                    idx = np.argsort(ind[:, 1])
                    id_map = ind[idx, 0]
                    id_map += self.args.num_labeled_classes

                    # targets_new = targets <-- this is not deep copy anymore due to NumPy version change
                    targets_new = np.copy(targets)
                    for i in range(self.args.num_unlabeled_classes1):
                        targets_new[targets == i + self.args.num_labeled_classes] = id_map[i]
                    targets = targets_new

                else:
                    ind = ind[:self.args.num_unlabeled_classes2, :]
                    idx = np.argsort(ind[:, 1])
                    id_map = ind[idx, 0]
                    id_map += self.args.num_labeled_classes

                    # targets_new = targets <-- this is not deep copy anymore due to NumPy version change
                    targets_new = np.copy(targets)
                    for i in range(self.args.num_unlabeled_classes2):
                        targets_new[targets == i + self.args.num_labeled_classes] = id_map[i]
                    targets = targets_new

            preds = torch.from_numpy(preds)
            targets = torch.from_numpy(targets)
            correct = preds.eq(targets).float().sum(0)
            acc = float(correct / targets.size(0))
            print('Epoch {}, Test acc {:.4f}'.format(epoch, acc))

        if return_ind:
            return acc, ind

        else:
            return acc

    def sample_labeled_features(self, class_mean, class_sig):
        feats = []
        labels = []

        if self.args.dataset_name == 'cifar10':
            num_per_class = 20
        elif self.args.dataset_name == 'cifar100':
            num_per_class = 2
        else:
            num_per_class = 3

        for i in range(self.args.num_labeled_classes):
            dist = torch.distributions.Normal(class_mean[i], class_sig.mean(dim=0))
            this_feat = dist.sample((num_per_class,)).cuda()  # new API
            this_label = torch.ones(this_feat.size(0)).cuda() * i

            feats.append(this_feat)
            labels.append(this_label)

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0).long()

        return feats, labels

    def generate_center(self):
        device = self.args.device

        all_feat = []
        all_labels = []

        class_mean = torch.zeros(self.args.num_labeled_classes, 512).cuda()
        class_sig = torch.zeros(self.args.num_labeled_classes, 512).cuda()

        print('Extract Labeled Feature')
        for epoch in range(1):
            self.old_model.eval()
            for batch_idx, (x, label, idx) in enumerate(tqdm(self.labeled_train_loader)):
                x, label = x.to(device), label.to(device)
                output1, output2, feat = self.old_model(x)

                all_feat.append(feat.detach().clone().cuda())
                all_labels.append(label.detach().clone().cuda())

        all_feat = torch.cat(all_feat, dim=0).cuda()
        all_labels = torch.cat(all_labels, dim=0).cuda()

        print('Calculate Labeled Mean-Var')
        for i in range(self.args.num_labeled_classes):
            this_feat = all_feat[all_labels == i]
            this_mean = this_feat.mean(dim=0)
            this_var = this_feat.var(dim=0)
            class_mean[i, :] = this_mean
            class_sig[i, :] = (this_var + 1e-5).sqrt()
        print('Finish')
        class_mean, class_sig, class_cov = class_mean.cuda(), class_sig.cuda(), 0  # class_cov.cuda()

        return class_mean, class_sig, class_cov
