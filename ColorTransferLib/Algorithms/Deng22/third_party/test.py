import torch
import torch.nn as nn
from torchvision import transforms
from .models import transformer
from .models import StyTR
import numpy as np
import cv2


def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

  
def predict(model_file_paths, src_img, ref_img, opt):
    # parser = argparse.ArgumentParser()
    # # Basic options
    # parser.add_argument('--content', type=str,
    #                     help='File path to the content image')
    # parser.add_argument('--content_dir', type=str,
    #                     help='Directory path to a batch of content images')
    # parser.add_argument('--style', type=str,
    #                     help='File path to the style image, or multiple style \
    #                     images separated by commas if you want to do style \
    #                     interpolation or spatial control')
    # parser.add_argument('--style_dir', type=str,
    #                     help='Directory path to a batch of style images')
    # parser.add_argument('--output', type=str, default='output',
    #                     help='Directory to save the output image(s)')
    # parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
    # parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
    # parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
    # parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')


    # parser.add_argument('--style_interpolation_weights', type=str, default="")
    # parser.add_argument('--a', type=float, default=1.0)
    # parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
    #                         help="Type of positional embedding to use on top of the image features")
    # parser.add_argument('--hidden_dim', default=512, type=int,
    #                         help="Size of the embeddings (dimension of the transformer)")
    # args = parser.parse_args()

    original_shape = src_img.shape

    # Advanced options
    content_size=512
    style_size=512
    crop='store_true'
    save_ext='.jpg'
    # output_path=args.output
    preserve_color='store_true'
    alpha=opt.a


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if device.type == "cuda":
        try:
            cuda_device_index = torch.cuda.current_device()
            print(f"Using CUDA device {cuda_device_index}: {torch.cuda.get_device_name(cuda_device_index)}")
        except AssertionError as e:
            print(f"Invalid CUDA device id: {e}")
            device = torch.device("cpu")
            print("Falling back to CPU.")
    else:
        print("CUDA is not available. Using CPU.")



    # Either --content or --content_dir should be given.
    # if args.content:
    #     content_paths = [Path(args.content)]
    # else:
    #     content_dir = Path(args.content_dir)
    #     content_paths = [f for f in content_dir.glob('*')]

    # # Either --style or --style_dir should be given.
    # if args.style:
    #     style_paths = [Path(args.style)]    
    # else:
    #     style_dir = Path(args.style_dir)
    #     style_paths = [f for f in style_dir.glob('*')]

    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)


    vgg = StyTR.vgg
    # vgg.load_state_dict(torch.load(args.vgg))
    vgg.load_state_dict(torch.load(model_file_paths["vgg_normalised.pth"]))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    # state_dict = torch.load(args.decoder_path)
    state_dict = torch.load(model_file_paths["decoder_iter_160000.pth"])
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    decoder.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    # state_dict = torch.load(args.Trans_path)
    state_dict = torch.load(model_file_paths["transformer_iter_160000.pth"])
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    Trans.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    # state_dict = torch.load(args.embedding_path)
    state_dict = torch.load(model_file_paths["embedding_iter_160000.pth"])
    for k, v in state_dict.items():
        #namekey = k[7:] # remove `module.`
        namekey = k
        new_state_dict[namekey] = v
    embedding.load_state_dict(new_state_dict)

    network = StyTR.StyTrans(vgg,decoder,embedding,Trans)
    network.eval()
    network.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    # for content_path in content_paths:
    #     for style_path in style_paths:
            # print(content_path)
        
            # content_tf1 = content_transform()       
            # content = content_tf(Image.open(content_path).convert("RGB"))

            # h, w, c = np.shape(content)    
            # style_tf1 = style_transform(h, w)
            # style = style_tf(Image.open(style_path).convert("RGB"))

            # style = style.to(device).unsqueeze(0)
            # content = content.to(device).unsqueeze(0)

    src_img = cv2.resize(src_img, dsize=(content_size,content_size))
    content = torch.from_numpy(np.transpose(src_img, (2, 0, 1))).float().unsqueeze(0)

    
    ref_img = cv2.resize(ref_img, dsize=(style_size, style_size))
    style = torch.from_numpy(np.transpose(ref_img, (2, 0, 1))).float().unsqueeze(0)

    

    with torch.no_grad():
        output = network(content, style)

    
    # Unpack the output tuple if necessary
    if isinstance(output, tuple):
        output = output[0]
    
    output = output.cpu()

    output = output.data.float().clamp_(0, 1).numpy()
    output = output.squeeze()
    output = np.transpose(output, (1, 2, 0))
    output_img = (output * 255.0).astype(np.uint8)
    output_img = cv2.resize(output_img, (original_shape[1], original_shape[0]))
    output_img = output_img.astype(np.float32)

    return output_img


