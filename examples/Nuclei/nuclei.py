import numpy as np
import imageio
import os
import cv2

class NucleiDataset(object):
    def __init__(self, root_dir, mode=None):
        assert mode in ['train', 'val', 'test']
        
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        
        # add data
        self.add_nuclei(root_dir, mode, split_ratio=0.9)

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
     
    def add_nuclei(self, root_dir, mode, split_ratio=0.9):
        # Add classes
        self.add_class("nuclei", 1, "nuclei") #source, id, name. id = 0s is BG
        
        image_names = os.listdir(root_dir)
        length = len(image_names)
        
        np.random.seed(1000)
        image_names = list(np.random.permutation(image_names))
        np.random.seed(None)
        
        if mode == 'train':
            image_names = image_names[: int(split_ratio*length)]
        if mode == 'val':
            image_names = image_names[int(split_ratio*length):]
        if mode == 'val_as_test':
            image_names = image_names[int(split_ratio*length):]     
            mode = 'test'
        dirs = [root_dir + img_name + '/images/' for img_name in image_names]
        mask_dirs = [root_dir + img_name + '/masks/' for img_name in image_names]

        # Add images
        for i in range(len(image_names)):
            self.add_image(
                source = "nuclei", 
                image_id = i,
                path = dirs[i] + image_names[i] + '.png',
                mask_dir = mask_dirs[i],
                name = image_names[i]
                )
      

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = imageio.imread(self.image_info[image_id]['path'])
        # RGBA to RGB
        if image.shape[2] != 3:
            image = image[:,:,:3]
        return image
    
    def load_image2(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = cv2.imread(self.image_info[image_id]['path'], cv2.IMREAD_COLOR)
        # RGBA to RGB
        if image.shape[2] != 3:
            image = image[:,:,:3]
        return image

    def image_reference(self, image_id):
        """Return the details of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            return info["path"]
        else:
            super(NucleiDataset, self).image_reference(self, image_id)

    def load_mask(self, image_id):
        """ 
        Returns:
            masks: A binary array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        mask_dir= info['mask_dir'] 
        mask_names = os.listdir(mask_dir)
        mask_paths = [mask_dir + mask_name for mask_name in mask_names]
        
        count = len(mask_paths)
        
        masks = [imageio.imread(path) for path in mask_paths]
        mask = np.stack(masks, axis=-1)
#        mask = mask.astype(bool)
        mask = np.where(mask>128, 1, 0)
        
        class_ids = np.ones(count,dtype=np.int32)
        return mask, class_ids


if __name__ == "__main__":
    ds = NucleiDataset('data/stage1_train/','train')
    #ds.add_nuclei('data/stage1_train/','train')
    print(len(ds.image_info))
    print(ds.image_info[0])
    
    image = ds.load_image(0)
    print(image.shape)
    
    image = ds.load_image2(0)
    print(image.shape)
    
    mask, _ = ds.load_mask(0)
    print(len(_))
    print(mask.shape)
    