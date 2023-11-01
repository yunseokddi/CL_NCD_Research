import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils.utils import AverageMeter, PairEnum
from utils import ramps


class FirstTrainer(object):
    def __init__(self, args, model, model_without_ddp, old_model, labeled_train_loader, mix_train_loader, optimizer,
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

    def train(self):
        if self.args.labeled_center > 0:
            self.class_mean, self.class_sig, self.class_cov = self.generate_center()
        else:
            self.class_mean, self.class_sig, self.class_cov = None, None, None

        for epoch in range(self.args.epochs):
            self._train_epoch(epoch)

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
            x, x_bar, label = x.to(self.args.device, non_blocking=True), x_bar.to(self.args.device,
                                                                                  non_blocking=True), label.to(
                self.args.device, non_blocking=True)

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
                'Total loss': loss_record.avg.item(),
                'CE loss': loss_ce_add_record.avg.item(),
                'BCE loss': loss_bce_record.avg.item(),
                'MSE loss' : consistency_loss_record.avg.item(),
                'KD loss' : loss_kd_record.avg.item()
            }

            tq_train.set_postfix(errors)


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
