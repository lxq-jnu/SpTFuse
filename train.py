from create_dataset import *
from utils import *
from network import SpTFuse
from options import * 
from saver import Saver, resume
from tqdm import tqdm
from optimizer import Optimizer
from mobile_sam import sam_model_registry,SamAutomaticMaskGenerator
from models.common import res_scale
from losses import Fusion_loss
import torch.distributed as dist
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())

    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main():
    parser = TrainOptions()
    args = parser.parse()

    # args.distributed = True
    args.distributed = False
    init_distributed_mode(args)
    if args.distributed:
        model = SpTFuse()
        model.cuda()
        SpTFuse_model = nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu],find_unused_parameters=True)
    else:
        SpTFuse_model = SpTFuse()
    SpTFuse_model.cuda()

    #SAM
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.cuda()
    amg_kwargs = get_amg_kwargs(args)
    seg_model = SamAutomaticMaskGenerator(sam, **amg_kwargs)

    # define dataset
    train_dataset = MSRS_Data(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            sampler = train_sampler,
            shuffle=(train_sampler is None),
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers = args.nThreads,
            )
    else:
        train_loader = torch.utils.data.DataLoader(
            shuffle=True,
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.nThreads,
        )


    ep_iter = len(train_loader)
    max_iter = args.n_ep * ep_iter
    print('Training iter: {}'.format(max_iter))
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-3
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5

    optimizer = Optimizer(
            model = SpTFuse_model,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

    if args.resume:
        SpTFuse_model, optimizer.optim, ep, total_it = resume(SpTFuse_model, optimizer.optim, args.resume)
        optimizer = Optimizer(
            model = SpTFuse_model,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power, 
            it=total_it)
        lr = optimizer.get_lr()
        print('lr:{}'.format(lr))
    else: 
        ep = -1
        total_it = 0
    ep += 1

    log_dir = os.path.join(args.display_dir, 'logger', args.name)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = logger_config(log_path=log_path, logging_name='Timer')

    trainer(train_loader,
            SpTFuse_model,
            seg_model,
            optimizer,
            args,
            logger,
            ep,
            total_it,
            args,
            train_sampler if args.distributed else None)
    
def trainer(train_loader,  SpTFuse_model, seg_model, optimizer, opt, logger, start_ep=0, total_it=0,args=None,train_sampler=None):
    total_epoch = opt.n_ep
    saver = Saver(opt)
    t = 50
    for ep in range(start_ep, total_epoch):
        SpTFuse_model.train()  
        if args.distributed:
            train_sampler.set_epoch(ep)      

        train_tqdm = tqdm(train_loader, total=len(train_loader))
        for img_vi, vi_Y, _, _, img_ir,mask, _ in train_tqdm:
            total_it += 1
            img_vi = img_vi.cuda()
            vi_Y = vi_Y.cuda()
            img_ir = img_ir.cuda()
            mask = mask.cuda()
            _,cc,H,W = img_vi.size()
            if cc == 1:
                img_vi = torch.cat([img_vi, img_vi, img_vi], dim=1)  

            with torch.no_grad():
                seg_model.predictor.set_image(img_vi)
                vis_input_image = seg_model.predictor.input_image
                vis_mid_out = seg_model.predictor.features
                _,C,_,_ = vis_mid_out.size()
                vis_padh = seg_model.predictor.padh
                vis_padw = seg_model.predictor.padw
                vis_scale = seg_model.predictor.scale
                vis_mid_res=[]
                for i in range(len(vis_mid_out)):
                    vis_res = res_scale(vis_input_image[i],vis_mid_out[i],vis_padh[i],vis_padw[i],vis_scale[i])
                    vis_mid_res.append(vis_res)
                vis_mid_out_res = torch.cat(vis_mid_res)

                img_ir_ = torch.cat([img_ir, img_ir, img_ir], dim=1)  
                seg_model.predictor.set_image(img_ir_)
                ir_input_image = seg_model.predictor.input_image
                ir_mid_out = seg_model.predictor.features
                ir_padh = seg_model.predictor.padh
                ir_padw = seg_model.predictor.padw
                ir_scale = seg_model.predictor.scale
                ir_mid_res=[]
                for i in range(len(vis_mid_out)):
                    ir_res = res_scale(ir_input_image[i],ir_mid_out[i],ir_padh[i],ir_padw[i],ir_scale[i])
                    ir_mid_res.append(ir_res)
                ir_mid_out_res = torch.cat(ir_mid_res)
            fused_img,vis_ploss,ir_ploss= SpTFuse_model(img_vi, img_ir,vis_mid_out_res,ir_mid_out_res,H,W,C)
            optimizer.zero_grad()
            t1, t2, t3 = eval(opt.loss_weight)
            fusion_loss,prio_loss, grad_loss, int_loss = Fusion_loss(vi_Y, img_ir, fused_img,vis_ploss,ir_ploss, mask=mask,weights=[t1,t2,t3])
            train_tqdm.set_postfix(epoch=ep,
                                   prior_loss = t1 * prio_loss.item(),
                                   gradinet_loss=t2 * grad_loss.item(),
                                   int_loss=t3 * int_loss.item(),
                                   loss_total=fusion_loss.item())
            fusion_loss.backward()
            optimizer.step()

        if ep % t == 0:
            saver.write_model(ep, total_it, SpTFuse_model, optimizer.optim, is_best=False)

if __name__ == '__main__':
    main()
