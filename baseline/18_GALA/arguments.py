import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3, help='Number of graph convolution layers')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=128, help='Dimension of graph convolution layers')
    parser.add_argument('--aug', type=str, default='random4')
    parser.add_argument('--randperm', type=int, default=1)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--log_dir', default='log_dir', help='directory to save log')
    parser.add_argument('--log_file', type=str, default='results.txt', help='name of file for logging')
    parser.add_argument('--start_upd_epoch', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=2)
    # GALA's graph encoder is a GCN.  Keep the argument for compatibility
    # with the original TU-dataset entry point, but default to the paper's
    # encoder and reject non-GCN encoders in the bailA runner.
    parser.add_argument('--conv_type', type=str, default='GCN')
    parser.add_argument('--use_bn', type=int, default=1)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--global_pool', type=str, default='sum')
    parser.add_argument('--mmd_filter_ratio', type=float, default=0.7)
    parser.add_argument('--aug_strength', type=float, default=0.1)
    parser.add_argument('--num_aug', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--tta_epoch', type=int, default=10)
    parser.add_argument('--data_split', dest='data_split', type=int, default=4,)
    parser.add_argument('--source_index', dest='source_index', type=int, default=0,)
    parser.add_argument('--target_index', dest='target_index', type=int, default=1,)
    parser.add_argument('--confident_percentage', type=float, default=0.5)
    parser.add_argument('--jigsaw', type=bool, default=False)

    # Settings for single-graph fairness source-free adaptation experiments.
    # Source/target domains are inferred from the dataset configuration unless
    # explicitly overridden; the default experiment seeds are 1 through 5.
    parser.add_argument('--source-domain', type=str, default=None)
    parser.add_argument('--target-domain', type=str, default=None)
    parser.add_argument('--data-root', type=str, default='dataset')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5',
                        help='Comma-separated seeds used by the fairness runner.')
    parser.add_argument('--source-epochs', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--adapt-ratio', type=float, default=0.8,
                        help='Unlabeled target nodes used for adaptation.')
    parser.add_argument('--diffusion-steps', type=int, default=20)
    parser.add_argument('--diffusion-noise', type=float, default=0.10)
    parser.add_argument('--pseudo-threshold', type=float, default=0.95)
    parser.add_argument('--min-pseudo-threshold', type=float, default=0.80)
    parser.add_argument('--pseudo-weight', type=float, default=1.0)
    parser.add_argument('--consistency-weight', type=float, default=1.0)
    parser.add_argument('--jigsaw-ratio', type=float, default=0.10)
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help=(
            'Execution device: auto, cpu, cuda, or cuda:N. '
            'Use --gpu-id with auto/cuda to select a CUDA device.'
        ),
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='Zero-based CUDA device index used by --device auto or --device cuda.',
    )

    return parser.parse_args()
