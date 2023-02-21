import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import numpy as np

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, num_classes):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    id_loss_meter = AverageMeter()
    id_losses_meters = [AverageMeter() for i in range(cfg.MODEL.CLS_TOKEN_NUM)]
    triplet_loss_meter = AverageMeter()
    dissimilar_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    scaler = amp.GradScaler()
    # train
    if cfg.MODEL.ID_HARD_MINING:
        weight = torch.normal(0, 1, size=(num_classes, 768))
    for epoch in range(1, epochs + 1):
        if cfg.MODEL.ID_HARD_MINING:
            train_loader.batch_sampler.sampler.update_weight(weight)
        start_time = time.time()
        loss_meter.reset()
        id_loss_meter.reset()
        for i in range(cfg.MODEL.CLS_TOKEN_NUM):
            id_losses_meters[i].reset() 
        triplet_loss_meter.reset()    
        dissimilar_loss_meter.reset()   
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat, weight = model(img, target, cam_label=target_cam, view_label=target_view )
                losses = loss_fn(score, feat, target, target_cam)
            id_loss = sum(losses[0]) / len(losses[0])
            tri_loss = sum(losses[1]) / len(losses[1])
            if cfg.MODEL.CLS_TOKENS_LOSS:
                loss = id_loss + tri_loss + losses[2]
            else:
                loss = id_loss + tri_loss

            scaler.scale(loss).backward()

            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            try:
                if isinstance(score, list):
                    if isinstance(score[0], tuple):
                        acc = (score[0][0].max(1)[1] == score[0][1]).float().mean()
                    else:
                        acc = (score[0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()
            except:
                acc = 0

            loss_meter.update(loss.item(), img.shape[0])
            id_loss_meter.update(id_loss.item(), img.shape[0])
            for i in range(cfg.MODEL.CLS_TOKEN_NUM):
                id_losses_meters[i].update(losses[0][i].item(), img.shape[0])
            triplet_loss_meter.update(tri_loss.item(), img.shape[0])
            if cfg.MODEL.CLS_TOKEN_NUM > 1:
                dissimilar_loss_meter.update(losses[2].item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                id_losses_avgs = [id_losses_meter.avg for id_losses_meter in id_losses_meters]
                id_losses_avgs_info = "{:.3f}".format(id_losses_avgs[0])
                for i in range(1, cfg.MODEL.CLS_TOKEN_NUM):
                    id_losses_avgs_info += "/{:.3f}".format(id_losses_avgs[i])
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID_Loss: {:.3f}-{}, TRIPLE_Loss: {:.3f}, DISSIMILAR Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, id_loss_meter.avg, id_losses_avgs_info, triplet_loss_meter.avg, dissimilar_loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, distmat, pids, camids, feats = evaluator.compute()
                    with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT), 'wb') as f:
                        np.save(f, distmat)
                        np.save(f, pids)
                        np.save(f, camids)
                        np.save(f, feats)
                    logger.info("dist_mat saved at: {}".format(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT)))
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, distmat, pids, camids, feats = evaluator.compute()
                with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT), 'wb') as f:
                    np.save(f, distmat)
                    np.save(f, pids)
                    np.save(f, camids)
                    np.save(f, feats)
                logger.info("dist_mat saved at: {}".format(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT)))
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, pids, camids, feats = evaluator.compute()
    with open(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT), 'wb') as f:
        np.save(f, distmat)
        np.save(f, pids)
        np.save(f, camids)
        np.save(f, feats)
    logger.info("dist_mat saved at: {}".format(os.path.join(cfg.OUTPUT_DIR, cfg.TEST.DIST_MAT)))
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


