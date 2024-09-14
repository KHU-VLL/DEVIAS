import os
from torchvision import transforms
from .kinetics import VideoClsDataset
from .ssv2 import SSVideoClsDataset
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import csv
import torch.nn.functional as F

from .hat_decode import VideoHATDataset
from .hvu import VideoClsDataset_HVU
from .activitynet import VideoClsDataset_ActivityNet

def is_directory_exists(path):
    return os.path.exists(path) and os.path.isdir(path)

def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
            data_path = os.path.join(args.data_prefix, 'videos_train')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
            data_path = os.path.join(args.data_prefix, 'videos_val')
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
            data_path = os.path.join(args.data_prefix, 'videos_val')

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)

        nb_classes = 400

    elif args.data_set == 'Kinetics-BG':
        if is_train :
            mode = 'train'
        elif test_mode :
            mode = 'test'
        else :
            mode = 'validation'

        anno_path = args.data_path
        data_path = args.data_prefix
        
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes

    elif args.data_set == 'UCF101-BG':
        if is_train :
            mode = 'train'
        elif test_mode :
            mode = 'test'
        else :
            mode = 'validation'

        anno_path = args.data_path
        data_path = args.data_prefix
        
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes
    
    elif args.data_set == 'Kinetics-HAT':
        if is_train :
            mode = 'train'
        elif test_mode :
            mode = 'test'
        else :
            mode = 'validation'
        
        anno_path = args.data_path
        data_path = args.data_prefix
        
        dataset = VideoHATDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes
        
    elif args.data_set == 'UCF101-HAT':
        if is_train :
            mode = 'train'
        elif test_mode :
            mode = 'test'
        else :
            mode = 'validation'
        
        anno_path = args.data_path
        data_path = args.data_prefix
        
        dataset = VideoHATDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = args.nb_classes
        
    elif args.data_set == 'Diving-48':
        mode = None
        anno_path = None 
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        data_path = args.data_prefix
            
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path=data_path,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                args=args)
        nb_classes = 48
        
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        data_path = args.data_prefix

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        # nb_classes = 174
        nb_classes = 87     # for mini_ssv2
        
    elif args.data_set in ['SCUBA']:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        
        data_path = args.data_prefix

        # data_root_dir_path like data/ucf101_places or data/kinetics_sinusoidal        
        assert 'places' in data_path or 'vqgan' in data_path or 'sinusoidal' in data_path
        assert 'ucf' in data_path or 'kinetics' in data_path in data_path
        
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        
        if 'ucf' in data_path :
            nb_classes = 101
        elif 'kinetics' in data_path :
            nb_classes = 400
        
    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        
        data_path = args.data_prefix
        
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        split = 1 
        
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, f'train{split}.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, f'test{split}.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, f'test{split}.csv') 
        data_path = args.data_prefix
        
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
        
        
    elif args.data_set == 'HVU':
        mode = None
        anno_path = None
        data_path = args.data_prefix
        
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
            data_path = os.path.join(data_path, 'train')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val_seen.csv') 
            data_path = os.path.join(data_path, 'val')
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val_seen.csv') 
            data_path = os.path.join(data_path, 'val')


        dataset = VideoClsDataset_HVU(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        
        nb_classes = 400
        
    elif args.data_set == 'HVU-EVAL':
        mode = 'validation'
        seen_comb_list, unseen_comb_list = args.anno_path

        data_path = os.path.join(args.data_prefix, 'val')
        dataset_lst = []

        for anno_path in [seen_comb_list, unseen_comb_list]:
            dataset_lst.append(VideoClsDataset_HVU(
                anno_path=anno_path,
                data_path=data_path,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                args=args))
            
        return dataset_lst
    
    elif args.data_set == 'ActivityNet':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        
        data_path = args.data_prefix
        
        dataset = VideoClsDataset_ActivityNet(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        
        nb_classes = 200

    else:
        raise NotImplementedError()
    
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes



def knn_build_dataset(is_train, args):
    if args.data_set == 'Places365':
        mode = None
        anno_path = None
        
        if is_train is True:
            anno_path = os.path.join(args.data_path, 'train.csv')
        else:  
            anno_path = os.path.join(args.data_path, 'val.csv') 
        data_path = args.data_prefix    # 'places365/val_256'
            
        nb_classes = 365
        dataset = PlacesDataset(csv_file=anno_path,                                 
                                root_dir=data_path)
            
    elif args.data_set == 'Diving-48':
        mode = None
        anno_path = None
        
        if is_train is True:
            anno_path = os.path.join(args.data_path, 'train.csv')
        else:  
            anno_path = os.path.join(args.data_path, 'val.csv') 
        mode = 'validation'
        
        data_path = args.data_prefix    # 'Diving48/rgb'
        
        test_mode = False
        dataset = VideoClsDataset(
                anno_path=anno_path,
                data_path=data_path,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                args=args)
        nb_classes = 48
        
    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        
        if is_train is True:
            anno_path = os.path.join(args.data_path, 'train.csv')
        else:  
            anno_path = os.path.join(args.data_path, 'val.csv') 
        mode = 'validation'
            
        data_path = args.data_prefix 

        test_mode = False
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
                
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        
        if is_train is True:
            anno_path = os.path.join(args.data_path, 'train1.csv')
        else:  
            anno_path = os.path.join(args.data_path, 'test1.csv') 
        mode = 'validation'
            
        data_path = args.data_prefix 

        test_mode = False
        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51     
        
    else:
        raise NotImplementedError()
    
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


# for Places365 
class PlacesDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.label_array = []

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            self.file_list = []
            for row in reader:
                label = int(row[1]) 
                self.label_array.append(label)
                self.file_list.append(row[0])
                
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            InflateImageToTensor(16),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')

        label = int(self.label_array[idx])

        image = self.transform(image)
        image = image.permute(1, 0, 2, 3)

        return image, label,idx
    

class InflateImageToTensor(object):
    def __init__(self, num_frames=16):
        self.num_frames = num_frames

    def __call__(self, img):
        tensor = F.to_tensor(img)
        return tensor.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
