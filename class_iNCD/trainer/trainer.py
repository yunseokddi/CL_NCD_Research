import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils.utils import AverageMeter
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
            self.class_mean_old, self.class_sig_old, self.class_cov_old = self.generate_center()
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
