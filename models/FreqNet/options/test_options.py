from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """
    Options used during testing. Extends BaseOptions with test-specific arguments.
    """
    
    def initialize(self, parser):
        """
        Define test-specific options and set training flag to False.

        Args:
            parser: ArgumentParser object to which arguments are added.

        Returns:
            parser: Updated ArgumentParser with test options.
        """
        
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--dataroot')
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')

        self.isTrain = False
        return parser
