import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ParserArguments():
    args = argparse.ArgumentParser()

    # Directory Setting 
    args.add_argument('--data_root', type=str, default='/data/workspace/vfuser/sungjoo/stargan-master/stargan-master/stargan_custom/train_image', help='dataset directory')
    args.add_argument('--exp', type=str, default='/data/workspace/vfuser/sungjoo/Binary_clf/exp/GE', help='output directory')

    # Hyperparameters Setting 
    args.add_argument('--train_val_ratio', type=float, default=0.9, help='# of dataset stratified split(train : valid)')
    args.add_argument('--nb_epoch', type=int, default=100, help='number of epochs (default=60)')
    args.add_argument('--batch_size', type=int, default=8, help='batch size (default=8)')
    
    # Pre-processing
    args.add_argument('--img_size', type=int, default=224, help='input size (default=224)')
    args.add_argument('--w_min', type=int, default=50, help='window min value (default=50)')
    args.add_argument('--w_max', type=int, default=180, help='window max value (default=180)')

    # Optimization Settings
    args.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate (default=5e-4)')
    args.add_argument('--optim', type=str, default='Adam', help='optimizer (default=SGD)')
    args.add_argument('--momentum', type=float, default=0.9, help='momentum (default=0.9)')
    args.add_argument('--wd', type=float, default=3e-2, help='weight decay of optimizer (default=0.03)')
    args.add_argument('--bias_decay', action='store_true', help='apply weight decay on bias (default=False)')
    args.add_argument('--warmup_epoch', type=int, default=10, help='learning rate warm-up epoch (default=5)')
    args.add_argument('--min_lr', type=float, default=5e-6, help='minimum learning rate setting of cosine annealing (default=5e-6)')
    # args.add_argument('--class_weights', type=str, default='1,4,6,9', help='class weights for loss function (default=1,4,6,9)')

    # Network
    args.add_argument('--network', type=str, default='efficientnet-b4', help='classifier network (default=resnet34)')
    args.add_argument('--resume', type=str, default='', help='resume pre-trained weights')
    args.add_argument('--gan_pth', type=str, default='pth/GAN/generator-291.pkl', help='resume pre-trained weights')
    args.add_argument('--dropout', type=float, default=0.5, help='dropout rate of FC layer (default=0.5)')

    # Augmentation
    args.add_argument('--augmentation', type=str, default='light', help="apply light or heavy augmentation (default=light)")          
    args.add_argument('--rot_factor', type=float, default=15, help='max rotation degree of augmentation (default=15)')
    args.add_argument('--scale_factor', type=float, default=0.15, help='max scaling factor of augmentation (default=0.15)')

    # Resource option
    args.add_argument('--use_gpu', default="True", type=str2bool, help='use gpu or not (cpu only)')
    args.add_argument('--gpu_id', default="0", type=str)

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args = args.parse_args()

    # GPU Setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    if args.use_gpu:
        args.ngpu = len(args.gpu_id.split(","))
    else:
        args.gpu_id = 'cpu'
        args.ngpu = 'cpu'    

    # Make Output Directory
    os.makedirs(args.exp, exist_ok=True)
    
    # Get Number of Class
    args.num_classes=2

    return args


def TestParserArguments():
    args = argparse.ArgumentParser()

    # Directory Setting 
    args.add_argument('--data_root', type=str, default='/home/vfuser/sungjoo/data/spine_data3/new_img_GE', help='data_patches ')
    # args.add_argument('--data_root', type=str, default='../../annotation/app/static/data_patches', help='data_patches ')
    args.add_argument('--exp', type=str, default='./test_exp', help='output directory')
    args.add_argument('--exp_root', type=str, default='./test_exp/', help='output directory')

    # Network
    args.add_argument('--network', type=str, default='efficientnet-b4', help='classifier network (default=resnet34)')
    args.add_argument('--resume', type=str, default='/home/vfuser/sungjoo/prediction_crop/exp/GE_5/epoch_057_val_loss_0.0868_val_f1_0.9582.pth', help='resume plant pre-trained weights')
    args.add_argument('--dropout', type=float, default=0.5, help='dropout rate of FC layer (default=0.5)')
    args.add_argument('--use_gpu', default="True", type=str2bool, help='use gpu or not (cpu only)')
    args.add_argument('--gpu_id', default="0,1,2,3", type=str)

    args.add_argument('--mode', type=str, default='test', help='submit일 때 test로 설정됩니다.')
    
    args = args.parse_args()

    # GPU Setting
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

    # Normalization target image

    if args.use_gpu:
        args.ngpu = len(args.gpu_id.split(","))
    else:
        args.gpu_id = 'cpu'
        args.ngpu = 'cpu'  

    # Make Output Directory
    args.exp = os.path.join(args.exp, 'test')
    os.makedirs(args.exp, exist_ok=True)
    
    args.num_classes=2

    return args
