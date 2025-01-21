from create_dataset import *
from utils import *
from network import SpTFuse
from options import *
from saver import resume, save_img_single
from tqdm import tqdm
from mobile_sam import SamAutomaticMaskGenerator,sam_model_registry
from models.common import res_scale
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if
                  v is not None}
    return amg_kwargs

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    SpTFuse_model = SpTFuse().cuda()
    SpTFuse_model = resume(SpTFuse_model, model_save_path=opts.resume, is_train=False)

    # 加载预训练分割模型的权重
    sam = sam_model_registry[opts.model_type](checkpoint=opts.checkpoint)
    _ = sam.cuda()
    amg_kwargs = get_amg_kwargs(opts)
    seg_model = SamAutomaticMaskGenerator(sam, **amg_kwargs)

    # define dataset
    test_dataset = FusionData(opts)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=opts.batch_size,
        shuffle=False)

    # Train and evaluate the network
    tester(test_loader, SpTFuse_model, opts, seg_model)


def tester(test_loader, SpTFuse_model, opts, seg_model):
    SpTFuse_model.eval()
    test_bar = tqdm(test_loader)
    ## define save dir
    Fusion_save_dir = os.path.join(opts.result_dir, opts.dataname)
    os.makedirs(Fusion_save_dir, exist_ok=True)
    with torch.no_grad():
        num_images = 0
        for it, (img_ir, img_vi, img_names, widths, heights) in enumerate(test_bar):
            num_images += len(img_names)
            img_vi = img_vi.cuda()
            img_ir = img_ir.cuda()
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
            vi_Y = vi_Y.cuda()
            vi_Cb = vi_Cb.cuda()
            vi_Cr = vi_Cr.cuda()

            B, C, H, W = img_vi.size()

            seg_model.predictor.set_image(img_vi)
            vis_input_image = seg_model.predictor.input_image
            vis_mid_out = seg_model.predictor.features
            _, C, _, _ = vis_mid_out.size()
            vis_padh = seg_model.predictor.padh
            vis_padw = seg_model.predictor.padw
            vis_scale = seg_model.predictor.scale
            vis_mid_res = []
            for i in range(len(vis_mid_out)):
                vis_res = res_scale(vis_input_image[i], vis_mid_out[i], vis_padh[i], vis_padw[i], vis_scale[i])
                vis_mid_res.append(vis_res)
            vis_mid_out_res = torch.cat(vis_mid_res)

            img_ir_ = torch.cat([img_ir, img_ir, img_ir], dim=1)
            seg_model.predictor.set_image(img_ir_)
            ir_input_image = seg_model.predictor.input_image
            ir_mid_out = seg_model.predictor.features
            ir_padh = seg_model.predictor.padh
            ir_padw = seg_model.predictor.padw
            ir_scale = seg_model.predictor.scale
            ir_mid_res = []
            for i in range(len(vis_mid_out)):
                ir_res = res_scale(ir_input_image[i], ir_mid_out[i], ir_padh[i], ir_padw[i], ir_scale[i])
                ir_mid_res.append(ir_res)
            ir_mid_out_res = torch.cat(ir_mid_res)
            fused_img,vis_ploss,ir_ploss = SpTFuse_model(img_vi, img_ir, vis_mid_out_res, ir_mid_out_res,H,W,C)

            fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
            for i in range(len(img_names)):
                img_name = img_names[i]
                fusion_save_name = os.path.join(Fusion_save_dir, img_name)
                save_img_single(fused_img[i, ::], fusion_save_name, widths[i], heights[i])
                test_bar.set_description('Image: {} '.format(img_name))

if __name__ == '__main__':
    main()
