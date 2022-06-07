# ██╗██╗███╗   ██╗████████╗
# ██║██║████╗  ██║╚══██╔══╝
# ██║██║██╔██╗ ██║   ██║
# ██║██║██║╚██╗██║   ██║
# ██║██║██║ ╚████║   ██║
# ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝

# @Time    : 2021-09-28 09:35:49
# @Author  : zhaosheng
# @email   : zhaosheng@nuaa.edu.cn
# @Blog    : iint.icu
# @File    : options/test_options.py
# @Describe: define options used during test time.

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str,
                            default='./results/', help='saves results here.')
        # parser.add_argument('--batch_size', type=int, default=1, help='')
        parser.add_argument('--aspect_ratio', type=float,
                            default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str,
                            default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true',default=True,
                            help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int,
                            default=9999, help='how many test images to run')
        parser.add_argument('--checkpoint', type=str, help='')
        
        # k-fold
        parser.add_argument('--k_type', type=str,
                            default="test", help='dont use k-fold')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.set_defaults(batch_size=1)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(display_id=-1)
        parser.set_defaults(phase="test")
        #parser.set_defaults(preprocess="none")
        parser.set_defaults(k_type="test")
        self.isTrain = False
        return parser
