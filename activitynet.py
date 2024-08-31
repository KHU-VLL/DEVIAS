import os
import numpy as np
import torch
from torchvision import transforms
from random_erasing import RandomErasing
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import os
import pandas as pd


class VideoClsDataset_ActivityNet(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,args=None,task_id =-1,loader='decord',rehearsal=False):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        self.loader = self.loadvideo_decord
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        self.label_array = []
        self.dataset_samples = []
        self.start_array = []
        self.end_array = []
        self.duration_array = []

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        self.dataset_samples = [os.path.join(self.data_path, ('v_'+str(i))) for i in cleaned.values[:, 0]]
        self.start_array = list(cleaned.values[:, 1])
        self.end_array = list(cleaned.values[:, 2])
        self.duration_array = list(cleaned.values[:, 3])
        self.label_array = list(cleaned.values[:, 4])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            video_info = self.dataset_samples[index]
            t_start = self.start_array[index]
            t_end = self.end_array[index]
            video_duration = self.duration_array[index]
            filename = self.dataset_samples[index]


            start_ratio= round(float(t_start) / float(video_duration),5)
            end_ratio= round(float(t_end) / float(video_duration),5)
            if end_ratio > 1:
                end_ratio = 1.0
            video_name = filename
            
            buffer = self.loader(video_name,start_ratio,end_ratio) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    print("video {} not correctly loaded during training".format(video_name))
                    index = np.random.randint(self.__len__())
                    
                    video_info = self.dataset_samples[index]
                    start_ratio= round(float(t_start) / float(video_duration),5)
                    end_ratio= round(float(t_end) / float(video_duration),5)
                    if end_ratio > 1:
                        end_ratio = 1.0
                    video_name = filename
            
                    buffer = self.loader(video_name,start_ratio,end_ratio) # T H W C

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            video_info = self.dataset_samples[index]
            t_start = self.start_array[index]
            t_end = self.end_array[index]
            video_duration = self.duration_array[index]
            filename = self.dataset_samples[index]


            start_ratio= round(float(t_start) / float(video_duration),5)
            end_ratio= round(float(t_end) / float(video_duration),5)
            if end_ratio > 1:
                end_ratio = 1.0

            video_name = filename

            buffer = self.loader(video_name,start_ratio,end_ratio) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    print("video {} not correctly loaded during validation".format(video_info))
                    index = np.random.randint(self.__len__())
                    video_info = self.dataset_samples[index]
                    start_ratio= round(float(t_start) / float(video_duration),5)
                    end_ratio= round(float(t_end) / float(video_duration),5)
                    if end_ratio > 1:
                        end_ratio = 1.0
                    video_name = filename
                    buffer = self.loader(video_name,start_ratio,end_ratio) # T H W C
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], video_name.split("/")[-1].split(".")[0]

        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

        buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True ,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


    def loadvideo_decord(self, video_name,start_ratio,end_ratio, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = os.path.join(self.data_path, video_name)

        extensions = ['mp4', 'mkv','webm']
        flag = False
        for ext in extensions:
            full_path = os.path.join(f"{fname}.{ext}")
            if os.path.exists(full_path):
                fname=full_path
                flag = True
                break
        if not flag:
            return []
        
        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        # handle temporal segments
        total_frames = len(vr) - 1

        start_frame = int(start_ratio * total_frames) 
        end_frame = int(end_ratio * total_frames) 
        video_length = end_frame - start_frame
        if video_length <= 0:
            print(f"Warning: video_length is zero or negative. Adjusting end_frame. {video_name}")
            video_length = 1

        average_duration = video_length // self.num_segment
        all_index = []
        if average_duration > 0:
            all_index += list(start_frame + np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration, size=self.num_segment))
        elif video_length > self.num_segment:
            all_index += list(start_frame + np.sort(np.random.randint(video_length, size=self.num_segment)))
        else:
            all_index += list(np.arange(start_frame, start_frame + self.num_segment) % video_length)
        all_index = list(np.array(all_index)) 
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()

        return buffer
        

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
