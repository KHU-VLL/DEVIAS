import os
import numpy as np
import torch
import decord
from PIL import Image
from torchvision import transforms
from utils.transform.random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import utils.transform.video_transforms as video_transforms 
import utils.transform.volume_transforms as volume_transforms
import random
from scipy import ndimage
import pickle

class VideoHATDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
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
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        anno_dir = "/".join(anno_path.split("/")[:-2])
        with open(os.path.join(anno_dir, "labels.csv"), 'r') as f :
            data = f.readlines()
        # idx,class_name
            
        label_ind = dict()
        for line in data :
            line = line.split(",")
            label_ind[line[1].replace("\n", "")] = int(line[0])
            
        if self.args.data_set == "Kinetics-HAT" :
            with open(anno_path, 'rb') as f :
                cleaned = pickle.load(f)

            self.dataset_samples, self.dataset_masks, self.background_len_array, self.label_array, self.dataset_inpaints = [], [], [], [], []
            for key_vid in list(cleaned.keys()) :
                contents = cleaned[key_vid]
                fg_class, fg_vid_name = key_vid.split('/')[0], key_vid.split('/')[1]
                fg_class_idx = label_ind[fg_class] 
                
                self.dataset_samples.append(os.path.join(self.data_path, 'original/videos', fg_vid_name))
                self.dataset_masks.append(os.path.join(self.data_path, 'seg/videos', fg_vid_name))
                self.dataset_inpaints.append(os.path.join(self.data_path, 'inpaint/videos', contents[0].split('/')[1]))
                self.background_len_array.append(contents[1])
                self.label_array.append(fg_class_idx)
                
        elif self.args.data_set == "UCF101-HAT" :
            with open(anno_path, 'rb') as f :
                cleaned = pickle.load(f)

            self.dataset_samples, self.dataset_masks, self.background_len_array, self.label_array, self.dataset_inpaints = [], [], [], [], []
            for key_vid in list(cleaned.keys()) :
                contents = cleaned[key_vid]
                fg_class, fg_vid_name = key_vid.split('/')[0], key_vid.split('/')[1]
                fg_class_idx = label_ind[fg_class] 
                
                self.dataset_samples.append(os.path.join(self.data_path, 'rawframes', key_vid))
                self.dataset_masks.append(os.path.join(self.data_path, 'seg', key_vid))
                self.dataset_inpaints.append(os.path.join(self.data_path, 'inpaint', contents[0]))
                self.background_len_array.append(int(contents[1]))
                self.label_array.append(fg_class_idx)

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
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_dataset_masks = []
            self.test_dataset_inpaints = []
            self.test_label_array = []
            self.test_background_len_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        if self.background_len_array is not None :
                            sample_video_len = self.background_len_array[idx]
                            self.test_background_len_array.append(sample_video_len)
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))
                        self.test_dataset_masks.append(self.dataset_masks[idx])
                        self.test_dataset_inpaints.append(self.dataset_inpaints[idx])
                            
    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            masks = self.dataset_masks[index]
            inpaints = self.dataset_inpaints[index]
            background_len = self.background_len_array[index]
            video_len = len(os.listdir(sample))

            buffer = self.loadvideo_frame(sample, video_len, background_len, masks, inpaints)
            if len(buffer) == 0:
                print(sample)
                print("load failed")
                exit(1)
                
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
                return frame_list, label_list, index_list, index
            else:
                buffer = self._aug_frame(buffer, args)
            
            return buffer, self.label_array[index], index, index

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            masks = self.dataset_masks[index]
            inpaints = self.dataset_inpaints[index]
            background_len = self.background_len_array[index]
            video_len = len(os.listdir(sample))

            buffer = self.loadvideo_frame(sample, video_len, background_len, masks, inpaints)
            if len(buffer) == 0:
                print(sample)
                print("load failed")
                exit(1)
                
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0], index

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            masks = self.test_dataset_masks[index]
            inpaints = self.test_dataset_inpaints[index]
            background_len = self.test_background_len_array[index]
            video_len = len(os.listdir(sample))
            
            dataset = self.args.data_set
            buffer = self.loadvideo_frame(sample, video_len, background_len, masks, inpaints, dataset)            

            while len(buffer) == 0:
                print(self.test_dataset[index])
                print(len(buffer))
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                exit(0)
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            if self.test_num_crop > 1:
                spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                    / (self.test_num_crop - 1)
            else:
                spatial_step = 0

            if self.test_num_segment > 1:
                temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                    / (self.test_num_segment - 1), 0)
            else:
                temporal_step = 0

            if self.test_num_crop == 1:
                spatial_start = (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) // 2
            else:
                spatial_start = int(split_nb * spatial_step)

            if self.test_num_segment == 1:
                temporal_start = (buffer.shape[0] - self.clip_len) // 2
            else:
                temporal_start = int(chunk_nb * temporal_step)

            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                    spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                    :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer) 
            #! buffer (C, T, H, W)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                chunk_nb, split_nb
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

    
    def loadvideo_frame(self, sample, video_len, background_len, mask_path, inpaint_path, dataset="HAT-frame") :
        #! load like decord        
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = video_len // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))
        
        bg_indices = np.linspace(1, background_len, video_len, endpoint=False).astype(int)
        bg_all_index = bg_indices[all_index]

        buffer = []
        for i, (frame_idx, bg_frame_idx) in enumerate(zip(all_index, bg_all_index)) :
            #! load frames
            if dataset == "UCF101-HAT" :
                frame_name = f"image_{str(frame_idx + 1).zfill(5)}.jpg"
            else :     
                frame_name = f"{str(frame_idx + 1).zfill(6)}.jpg"
            img = Image.open(os.path.join(sample, frame_name))
            img_short_size = img.size[1] if img.size[0] > img.size[1] else img.size[0]
            resize = transforms.Resize(img_short_size)
                        
            if dataset == "UCF101-HAT" :
                frame_name = f"img_{str(frame_idx).zfill(5)}.png"
            else :     
                frame_name = f"{str(frame_idx + 1).zfill(6)}.png"
            mask = Image.open(os.path.join(mask_path, frame_name)).convert('L').resize(img.size)
            mask = np.array(mask)
            
            if i == 0 :
                #! get movement
                if dataset == "UCF101-HAT" :
                    frame_name = f"img_{str(bg_frame_idx).zfill(5)}.png"
                    bg_mask = Image.open(os.path.join(inpaint_path.replace("inpaint/", "seg/"), frame_name)).convert('L')
                else :     
                    frame_name = f"{str(bg_frame_idx + 1).zfill(6)}.png"
                    bg_mask = Image.open(os.path.join(inpaint_path.replace("inpaint/", "seg/"), frame_name)).convert('L')
                bg_mask = np.array(resize(bg_mask))
                bg_mask = np.tile(np.expand_dims(bg_mask, axis=2), (1, 1, 3))
                
                fg_mask_ = np.tile(np.expand_dims(mask, axis=2), (1, 1, 3))
                if np.sum(fg_mask_) > 0 :
                    fg_center = ndimage.center_of_mass(fg_mask_)[:2]
                else :
                    fg_center = (fg_mask_.shape[0]/2, fg_mask_.shape[1]/2)
                    
                if np.sum(bg_mask) > 0 :
                    bg_center = ndimage.center_of_mass(bg_mask)[:2]
                else :
                    bg_center = (bg_mask.shape[0]/2, bg_mask.shape[1]/2)
                
                movement = (int(bg_center[0] - fg_center[0]), int(bg_center[1] - fg_center[1]))
            
            fg_mask = (mask > 128).astype(np.uint8)  
            fg_mask_img = Image.fromarray(fg_mask * 255).convert('L')

            if dataset == "UCF101-HAT" :
                frame_name = f"img_{str(bg_frame_idx).zfill(5)}.jpg"
            else :     
                frame_name = f"{str(bg_frame_idx + 1).zfill(6)}.jpg"
            inp = Image.open(os.path.join(inpaint_path, frame_name))  
            inp = resize(inp) 
            
            inp.paste(img, (movement[1], movement[0]), fg_mask_img)
            
            buffer.append(np.array(inp))
            
        return np.array(buffer)
    
    def loadvideo_frame_test(self, sample, video_len, background_len, mask_path, inpaint_path, chunk_nb, temporal_step) :
        #! get all frames with stride 
        all_index = [x for x in range(0, video_len, self.frame_sample_rate)]
        while len(all_index) < self.clip_len:
            all_index.append(all_index[-1])

        if self.test_num_segment == 1:
            temporal_start = (all_index.shape[0] - self.clip_len) // 2
        else:
            temporal_start = int(chunk_nb * temporal_step)

        all_index = all_index[temporal_start:temporal_start + self.clip_len]

        bg_indices = np.linspace(1, background_len, video_len, endpoint=False).astype(int)
        bg_all_index = bg_indices[all_index]

        buffer = []
        for i, (frame_idx, bg_frame_idx) in enumerate(zip(all_index, bg_all_index)) :
            #! load frames
            frame_name = f"{str(frame_idx + 1).zfill(6)}.jpg"
            img = Image.open(os.path.join(sample, frame_name))
            img_short_size = img.size[1] if img.size[0] > img.size[1] else img.size[0]
            resize = transforms.Resize(img_short_size)
                        
            frame_name = f"{str(frame_idx + 1).zfill(6)}.png"
            mask = Image.open(os.path.join(mask_path, frame_name)).convert('L').resize(img.size)
            fg_mask_img = mask
            mask = np.array(mask)
            fg_mask = (mask > 128).astype(np.uint8)  
            fg_mask = np.tile(np.expand_dims(fg_mask, axis=2), (1, 1, 3))
            
            if i == 0 :
                #! get movement
                frame_name = f"{str(bg_frame_idx + 1).zfill(6)}.png"
                bg_mask = Image.open(os.path.join(inpaint_path.replace("inp/", "seg/"), frame_name)).convert('L')
                bg_mask = np.array(resize(bg_mask))
                bg_mask = (bg_mask > 128).astype(np.uint8)  
                bg_mask = np.tile(np.expand_dims(bg_mask, axis=2), (1, 1, 3))
                
                if np.sum(fg_mask) > 0 :
                    fg_center = ndimage.center_of_mass(fg_mask)[:2]
                else :
                    fg_center = (fg_mask.shape[0]/2, fg_mask.shape[1]/2)
                    
                if np.sum(bg_mask) > 0 :
                    bg_center = ndimage.center_of_mass(bg_mask)[:2]
                else :
                    bg_center = (bg_mask.shape[0]/2, bg_mask.shape[1]/2)
                
                movement = (int(bg_center[0] - fg_center[0]), int(bg_center[1] - fg_center[1]))
            
            frame_name = f"{str(bg_frame_idx + 1).zfill(6)}.jpg"
            inp = Image.open(os.path.join(inpaint_path, frame_name))  
            inp = resize(inp) 
            
            inp.paste(img, (movement[1], movement[0]), fg_mask_img)
            
            buffer.append(np.array(inp))
            
        return np.array(buffer)
    
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


