# encoding: utf-8

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy, TemperatureCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .dissimilar_loss import Dissimilar


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    dissimilar = Dissimilar(dynamic_balancer=cfg.MODEL.DYNAMIC_BALANCER)
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    if cfg.MODEL.IF_TEMPERATURE_SOFTMAX == 'on':
        xent = TemperatureCrossEntropy()
        print("temperature softmax on")

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_TEMPERATURE_SOFTMAX == 'on':
                    if isinstance(score, list):
                        if isinstance(score[0], tuple):
                            ID_LOSS = [xent(scor, lbl) for scor, lbl in score]
                        else:
                            ID_LOSS = [xent(scor, target, t) for scor, t in zip(score, cfg.MODEL.TEMPERATURE)]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)
                else:
                    if isinstance(score, list):
                        if isinstance(score[0], tuple):
                            ID_LOSS = [F.cross_entropy(scor, lbl) for scor, lbl in score]
                        else:
                            ID_LOSS = [F.cross_entropy(scor, target) for scor in score]
                        # ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSSes = ID_LOSS
                    else:
                        ID_LOSS = F.cross_entropy(score, target)
                        ID_LOSSes = [ID_LOSS]

                if isinstance(feat, list):
                    TRI_LOSS = [triplet(feats, target)[0] for feats in feat]
                    # TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                    TRI_LOSSes = TRI_LOSS
                else:
                    TRI_LOSS = triplet(feat, target)[0]
                    TRI_LOSSes = [TRI_LOSS]

                if len(feat) > 1:
                    Dissimilar_LOSS = dissimilar(torch.stack(feat, dim=1))
                else:
                    Dissimilar_LOSS = 0

                return [[id_loss * cfg.MODEL.ID_LOSS_WEIGHT for id_loss in ID_LOSSes],
                            [tri_loss * cfg.MODEL.TRIPLET_LOSS_WEIGHT for tri_loss in TRI_LOSSes],
                                cfg.MODEL.DIVERSE_CLS_WEIGHT * Dissimilar_LOSS]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


