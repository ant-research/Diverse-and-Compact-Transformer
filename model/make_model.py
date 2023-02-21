import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss, PartSoftmax

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.ID_hardmining = cfg.MODEL.ID_HARD_MINING
        self.cls_token_num = cfg.MODEL.CLS_TOKEN_NUM
        self.feat_mean = cfg.TEST.MEAN_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, cfg=cfg)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        # if self.ID_LOSS_TYPE == 'arcface':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = Arcface(self.in_planes, self.num_classes,
        #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'cosface':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = Cosface(self.in_planes, self.num_classes,
        #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'amsoftmax':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = AMSoftmax(self.in_planes, self.num_classes,
        #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'circle':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = CircleLoss(self.in_planes, self.num_classes,
        #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # else:
        #     self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        #     self.classifier.apply(weights_init_classifier)
        # self.bottleneck = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck.bias.requires_grad_(False)
        # self.bottleneck.apply(weights_init_kaiming)
        if self.ID_LOSS_TYPE == 'partsoftmax':
            self.classifiers = nn.ModuleList([
            PartSoftmax(self.in_planes, self.num_classes, ratio=cfg.MODEL.PART_ID_RATIO)
            for i in range(self.cls_token_num)])
        else:
            self.classifiers = nn.ModuleList([
                nn.Linear(self.in_planes, self.num_classes, bias=False)
                for _ in range(self.cls_token_num)])
        self.bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(self.in_planes)
            for _ in range(self.cls_token_num)])
        for classifier, bottleneck in zip(self.classifiers, self.bottlenecks):
            if self.ID_LOSS_TYPE == 'softmax':
                classifier.apply(weights_init_classifier)
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
        
        if pretrain_choice == 'self':
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feats = self.base(x, cam_label=cam_label, view_label=view_label)
        feats = [bottleneck(global_feats[:, i]) for i, bottleneck in enumerate(self.bottlenecks)]

        if self.training:
            weight = None
            if self.ID_LOSS_TYPE == 'softmax':
                cls_scores = [classifier(feats[i]) for i, classifier in enumerate(self.classifiers)]
                if self.ID_hardmining:
                    weight = self.classifiers[0].weight.cpu()
            else:
                cls_scores = [classifier(feats[i], label) for i, classifier in enumerate(self.classifiers)]
            return cls_scores, [global_feats[:, i] for i in range(global_feats.shape[1])], weight
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.mean(torch.stack(feats, dim=1), dim=1) if self.feat_mean else torch.cat(feats, dim=1)
            else:
                # print("Test with feature before BN")
                return torch.mean(global_feats, dim=1) if self.feat_mean else global_feats.view(global_feats.shape[0], -1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.ID_hardmining = cfg.MODEL.ID_HARD_MINING
        self.cls_token_num = cfg.MODEL.CLS_TOKEN_NUM
        self.feat_mean = cfg.TEST.MEAN_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, cfg=cfg)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm

        self.local_blocks = nn.ModuleList([
            nn.Sequential(
                copy.deepcopy(block),
                copy.deepcopy(layer_norm)) for i in range(self.cls_token_num)
        ])

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        
        if self.ID_LOSS_TYPE == 'partsoftmax':
            self.classifiers = nn.ModuleList([
            PartSoftmax(self.in_planes, self.num_classes, ratio=cfg.MODEL.PART_ID_RATIO)
            for i in range(self.cls_token_num)])
        else:
            self.classifiers = nn.ModuleList([
                nn.Linear(self.in_planes, self.num_classes, bias=False)
                for _ in range(self.cls_token_num)])
        self.bottlenecks = nn.ModuleList([
            nn.BatchNorm1d(self.in_planes)
            for _ in range(self.cls_token_num)])
        for classifier, bottleneck in zip(self.classifiers, self.bottlenecks):
            if self.ID_LOSS_TYPE == 'softmax':
                classifier.apply(weights_init_classifier)
            bottleneck.bias.requires_grad_(False)
            bottleneck.apply(weights_init_kaiming)
        
        if pretrain_choice == 'self':
            self.load_param(model_path)

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        global_feats = torch.cat([self.local_blocks[i](
            torch.cat(
                (features[:, i:i+1], features[:, self.cls_token_num:]), dim=1)
            )[:, 0:1] for i in range(self.cls_token_num)], dim=1)
        feats = [bottleneck(global_feats[:, i]) for i, bottleneck in enumerate(self.bottlenecks)]

        if self.training:
            weight = None
            if self.ID_LOSS_TYPE == 'softmax':
                cls_scores = [classifier(feats[i]) for i, classifier in enumerate(self.classifiers)]
            else:
                cls_scores = [classifier(feats[i], label) for i, classifier in enumerate(self.classifiers)]
            return cls_scores, [global_feats[:, i] for i in range(global_feats.shape[1])], weight
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.mean(torch.stack(feats, dim=1), dim=1) if self.feat_mean else torch.cat(feats, dim=1)
            else:
                # print("Test with feature before BN")
                return torch.mean(global_feats, dim=1) if self.feat_mean else global_feats.view(global_feats.shape[0], -1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
