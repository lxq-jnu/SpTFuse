import argparse


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--dataroot', type=str, default=r'E:\dataset_metric\datasets\MSRS/', help='path of data')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--nThreads', type=int, default=0, help='# of threads for data loader')

        # training related
        self.parser.add_argument('--lr', default=1e-3, type=int, help='Initial learning rate for training model')
        self.parser.add_argument('--weight', default='dwa', type=str, help='multi-task weighting: equal, uncert, dwa')
        self.parser.add_argument('--n_ep', type=int, default=2000, help='number of epochs')
        self.parser.add_argument('--n_ep_decay', type=int, default=1000,
                                 help='epoch start decay learning rate, set -1 if no decay')
        # self.parser.add_argument('--resume', type=str, default="checkpoint/01700.pth",
        #                          help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=-1, help='GPU id')
        self.parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
        self.parser.add_argument('--loss_weight', default='[7, 155, 10]', type=str, metavar='N', help='loss weight')

         # ouptput related
        self.parser.add_argument('--name', type=str, default='SpTFuse', help='folder name to save outputs')
        self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='checkpoint',
                                 help='path for saving result images and models')
        self.parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=10, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=10, help='freq (epoch) of saving models')

        # SAM参数设置
        self.parser.add_argument(
            "--checkpoint",
            default="mobile_sam/weights/mobile_sam.pt",
            type=str,

            help="The path to the SAM checkpoint to use for mask generation.",
        )
        self.parser.add_argument(
            "--model-type",
            default="vit_t",
            type=str,

            help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
        )
        amg_settings = self.parser.add_argument_group("AMG Settings")

        amg_settings.add_argument(
            "--points-per-side",
            type=int,
            default=None,
            help="Generate masks by sampling a grid over the image with this many points to a side.",
        )

        amg_settings.add_argument(
            "--points-per-batch",
            type=int,
            default=None,
            help="How many input points to process simultaneously in one batch.",
        )

        amg_settings.add_argument(
            "--pred-iou-thresh",
            type=float,
            default=None,
            help="Exclude masks with a predicted score from the model that is lower than this threshold.",
        )

        amg_settings.add_argument(
            "--stability-score-thresh",
            type=float,
            default=None,
            help="Exclude masks with a stability score lower than this threshold.",
        )

        amg_settings.add_argument(
            "--stability-score-offset",
            type=float,
            default=None,
            help="Larger values perturb the mask more when measuring stability score.",
        )

        amg_settings.add_argument(
            "--box-nms-thresh",
            type=float,
            default=None,
            help="The overlap threshold for excluding a duplicate mask.",
        )

        amg_settings.add_argument(
            "--crop-n-layers",
            type=int,
            default=None,
            help=(
                "If >0, mask generation is run on smaller crops of the image to generate more masks. "
                "The value sets how many different scales to crop at."
            ),
        )

        amg_settings.add_argument(
            "--crop-nms-thresh",
            type=float,
            default=None,
            help="The overlap threshold for excluding duplicate masks across different crops.",
        )

        amg_settings.add_argument(
            "--crop-overlap-ratio",
            type=int,
            default=None,
            help="Larger numbers mean image crops will overlap more.",
        )

        amg_settings.add_argument(
            "--crop-n-points-downscale-factor",
            type=int,
            default=None,
            help="The number of points-per-side in each layer of crop is reduced by this factor.",
        )

        amg_settings.add_argument(
            "--min-mask-region-area",
            type=int,
            default=None,
            help=(
                "Disconnected mask regions or holes with area smaller than this value "
                "in pixels are removed by postprocessing."
            ),
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # data loader related
        self.parser.add_argument('--dataname', type=str, default='MSRS', help='name of dataset')
        self.parser.add_argument('--dataroot', type=str, default='E:\dataset_metric\datasets/', help='path of data')
        self.parser.add_argument('--phase', type=str, default='1', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--nThreads', type=int, default=16, help='# of threads for data loader')

        ## mode related
        self.parser.add_argument('--resume', type=str, default="checkpoint/01700.pth",
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')

        # results related
        self.parser.add_argument('--name', type=str, default='SpTFuse', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str, default='./msrs',
                                 help='path for saving result images and models')

        # SAM参数设置
        self.parser.add_argument(
            "--checkpoint",
            default="mobile_sam/weights/mobile_sam.pt",
            type=str,

            help="The path to the SAM checkpoint to use for mask generation.",
        )
        self.parser.add_argument(
            "--model-type",
            default="vit_t",
            type=str,

            help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
        )
        amg_settings = self.parser.add_argument_group("AMG Settings")

        amg_settings.add_argument(
            "--points-per-side",
            type=int,
            default=None,
            help="Generate masks by sampling a grid over the image with this many points to a side.",
        )

        amg_settings.add_argument(
            "--points-per-batch",
            type=int,
            default=None,
            help="How many input points to process simultaneously in one batch.",
        )

        amg_settings.add_argument(
            "--pred-iou-thresh",
            type=float,
            default=None,
            help="Exclude masks with a predicted score from the model that is lower than this threshold.",
        )

        amg_settings.add_argument(
            "--stability-score-thresh",
            type=float,
            default=None,
            help="Exclude masks with a stability score lower than this threshold.",
        )

        amg_settings.add_argument(
            "--stability-score-offset",
            type=float,
            default=None,
            help="Larger values perturb the mask more when measuring stability score.",
        )

        amg_settings.add_argument(
            "--box-nms-thresh",
            type=float,
            default=None,
            help="The overlap threshold for excluding a duplicate mask.",
        )

        amg_settings.add_argument(
            "--crop-n-layers",
            type=int,
            default=None,
            help=(
                "If >0, mask generation is run on smaller crops of the image to generate more masks. "
                "The value sets how many different scales to crop at."
            ),
        )

        amg_settings.add_argument(
            "--crop-nms-thresh",
            type=float,
            default=None,
            help="The overlap threshold for excluding duplicate masks across different crops.",
        )

        amg_settings.add_argument(
            "--crop-overlap-ratio",
            type=int,
            default=None,
            help="Larger numbers mean image crops will overlap more.",
        )

        amg_settings.add_argument(
            "--crop-n-points-downscale-factor",
            type=int,
            default=None,
            help="The number of points-per-side in each layer of crop is reduced by this factor.",
        )

        amg_settings.add_argument(
            "--min-mask-region-area",
            type=int,
            default=None,
            help=(
                "Disconnected mask regions or holes with area smaller than this value "
                "in pixels are removed by postprocessing."
            ),
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
