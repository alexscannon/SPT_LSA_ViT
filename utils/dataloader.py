import os
import random
import shutil
from colorama import Fore, Style
from torch.utils.data import Subset, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

def datainfo(logger, args):
    if args.dataset == 'CIFAR10':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR10')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        img_size = 32

    elif args.dataset == 'CIFAR100':
        print(Fore.YELLOW+'*'*80)
        logger.debug('CIFAR100')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 100
        img_mean, img_std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        img_size = 32

    elif args.dataset == 'SVHN':
        print(Fore.YELLOW+'*'*80)
        logger.debug('SVHN')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 10
        img_mean, img_std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        img_size = 32

    elif args.dataset == 'T-IMNET':
        print(Fore.YELLOW+'*'*80)
        logger.debug('T-IMNET')
        print('*'*80 + Style.RESET_ALL)
        n_classes = 200
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 64

    data_info = dict()
    data_info['n_classes'] = n_classes
    data_info['stat'] = (img_mean, img_std)
    data_info['img_size'] = img_size

    return data_info

def dataload(args, augmentations, normalize, data_info):
    class_info = {} # Will populate only for CIFAR100 situation

    if args.dataset == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=True,
            transform=augmentations
        )

        val_dataset = datasets.CIFAR10(
            root=args.data_path,
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.Resize(data_info['img_size']),
                transforms.ToTensor(),
                *normalize])
        )

    elif args.dataset == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            root=args.data_path,
            train=True,
            download=True,
            transform=augmentations
        )

        val_dataset = datasets.CIFAR100(
            root=args.data_path,
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.Resize(data_info['img_size']),
                transforms.ToTensor(),
                *normalize])
        )

        if args.ind_class_ratio < 1. or args.ind_example_ratio < 1.:
            train_dataset, class_info, val_dataset = trim_CIFAR100_dataset(
                args=args,
                n_classes=data_info['n_classes'],
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )
            print(f"CIFAR100 training dataset is reduced to {len(train_dataset)} examples from original 50,000...")
            print(f"CIFAR100 validation dataset is reduced to {len(val_dataset)} examples from original 10,000...")

    elif args.dataset == 'SVHN':
        train_dataset = datasets.SVHN(
            root=args.data_path,
            split='train',
            download=True,
            transform=augmentations
        )

        val_dataset = datasets.SVHN(
            root=args.data_path,
            split='test',
            download=True,
            transform=transforms.Compose([
                transforms.Resize(data_info['img_size']),
                transforms.ToTensor(),
                *normalize])
            )

    elif args.dataset == 'T-IMNET':
        # Set data paths for either full or trimmed training and validation sets
        if args.ind_class_ratio < 1. or args.ind_example_ratio < 1.:
            data_path_train, data_path_val, class_info = build_tinyimagenet_subset(args)

        else:
            data_path_train = os.path.join(args.data_path, 'tiny-imagenet-200', 'train')
            data_path_val = os.path.join(args.data_path, 'tiny-imagenet-200', 'val_organized')

        train_dataset = datasets.ImageFolder(
            root=data_path_train,
            transform=augmentations
        )

        val_dataset = datasets.ImageFolder(
            root=data_path_val,
            transform=transforms.Compose([
                transforms.Resize(data_info['img_size']),
                transforms.ToTensor(),
                *normalize])
        )

        print(f"Size of training data: {len(train_dataset)}...")
        print(f"Size of validation data: {len(val_dataset)}...")

    return train_dataset, val_dataset, class_info

# =================== MS Project Customization =================== #

class DatasetClassRemapping(Dataset):
    def __init__(self, dataset, class_remapping):
        self.dataset = dataset
        self.class_remapping = class_remapping

    def __getitem__(self, idx):
        img, old_label = self.dataset[idx]
        label = self.class_remapping[old_label]

        return img, label

    def __len__(self):
        return len(self.dataset)

def trim_CIFAR100_dataset(args, n_classes, train_dataset, val_dataset):
    """
    Takes the original CIFAR100 training dataset and trims it down for
    masters project continual learning objective. The split is usually
    80 classes for in-distribution and 75% of those 80 classes for IND
    training examples.
    """

    # Split datasets into train and val

    all_train_labels = np.array(train_dataset.targets) # array of each example's label [19, 0, 1, 66, 21]
    all_val_labels = np.array(val_dataset.targets) # array of each example's label [19, 0, 1, 66, 21]

    all_class_ids = np.unique(all_train_labels) # list of all unique possible targets in the dataset

    np.random.shuffle(all_class_ids) # Randomize class ordering
    print(f"Number of total dataset classes: {len(all_class_ids)}") # length should be 100

    # Compute IND and OOD class ids
    ind_class_ids = all_class_ids[:int(n_classes * args.ind_class_ratio)]
    ood_class_ids = all_class_ids[int(n_classes * args.ind_class_ratio):]

    # Create a mapping of the new target to old target {0: 21, 1: 44, ..., 99: 12}
    ind_class_mapping = {class_id: i for i, class_id in enumerate(ind_class_ids)}

    # TRAINING - For each IND class, grab associated examples, randomize, split into pretrain and left out examples
    train_ind_indices, left_out_ind_indices, ood_exmaple_idxs = [], [], []
    for ind_class in ind_class_ids:
        # Retrieve all examples in the dataset with the current class label
        class_example_idxs = np.where(all_train_labels == ind_class)[0].tolist()
        # randomize the examples
        np.random.shuffle(class_example_idxs)

        # Split the class-specific examples into what examples will be used
        train_ind_indices.extend(class_example_idxs[:int(len(class_example_idxs) * args.ind_example_ratio)])
        left_out_ind_indices.extend(class_example_idxs[int(len(class_example_idxs) * args.ind_example_ratio):])

    for ood_id in ood_class_ids:
        ood_exmaple_idxs.extend(np.where(all_train_labels == ood_id)[0].tolist())

    print(f"Number of training examples: {len(train_ind_indices)} ")
    print(f"Number of in-distribution left out examples: {len(left_out_ind_indices)} ")


    # VALIDATION - For each IND class, grab associated examples, randomize, split into pretrain and left out examples
    val_ind_indices = []
    for ind_class in ind_class_ids:
        # Retrieve all examples in the dataset with the current class label
        class_example_idxs = np.where(all_val_labels == ind_class)[0].tolist()
        # randomize the examples
        np.random.shuffle(class_example_idxs)
        # Split the class-specific examples into what examples will be used
        val_ind_indices.extend(class_example_idxs)

    print(f"Number of validation examples: {len(val_ind_indices)} ")

    # Store class information
    class_info = {
        'num_of_classes': n_classes * args.ind_class_ratio, # Old name for this property "n_classes"
        'pretrain_classes': ind_class_ids,
        'pretrained_ind_indices': train_ind_indices,
        'val_ind_indices': val_ind_indices,
        'left_out_classes': ood_class_ids, # Old name for this property "continual_classes"
        'left_out_ind_indices': left_out_ind_indices, # Old name for this property "left_out_indices"
        'pretrain_class_mapping': ind_class_mapping, # Old name for this property "class_mapping",
        'ood_example_idxs': ood_exmaple_idxs
    }

    train_dataset = DatasetClassRemapping(
        dataset=Subset(train_dataset, train_ind_indices),
        class_remapping=ind_class_mapping
    )

    val_dataset = DatasetClassRemapping(
        dataset=Subset(val_dataset, val_ind_indices),
        class_remapping=ind_class_mapping
    )

    return train_dataset, class_info, val_dataset




def build_tinyimagenet_subset(args):
    # Get tiny-imagenet datasets locations
    tiny_imagenet_dir = os.path.join(args.data_path, 'tiny-imagenet-200')

    train_dataset_path = os.path.join(tiny_imagenet_dir, 'train')
    val_dataset_path = os.path.join(tiny_imagenet_dir, 'val_organized')

    wnids_file = os.path.join(tiny_imagenet_dir, 'wnids.txt')

    # Set the save paths
    # Training bifercating of (IND vs. OOD) + (IND Left In vs. IND Left Out)
    save_path_train_ind_left_in = os.path.join(tiny_imagenet_dir, 'train_ind_in') # Save dir for left-in training in-distribution examples
    save_path_train_ind_left_out = os.path.join(tiny_imagenet_dir, 'train_ind_out') # Save dir for left-out training in-distribution examples
    save_path_train_ood = os.path.join(tiny_imagenet_dir, 'train_ood') # Save dir for training OOD examples

    # Validation bifercating of (IND vs. OOD). NO NEED for IND Left In vs. IND Left Out as its the validation set
    save_path_val_ind = os.path.join(tiny_imagenet_dir, 'val_ind') # Save dir for validation in-distribution examples
    save_path_val_ood = os.path.join(tiny_imagenet_dir, 'val_ood') # Save dir for validation OOD examples

    # Perform clean up – Remove any existing folders that might have different class or example splits
    new_save_paths = [save_path_train_ind_left_in, save_path_train_ind_left_out, save_path_train_ood, save_path_val_ind, save_path_val_ood]
    for path in new_save_paths:
        if os.path.exists(path):
            shutil.rmtree(path) # Recursively remove entire parent folder
        os.mkdir(path) # Create empty parent folder


    # Get all wnids
    with open(wnids_file, 'r') as f:
        wnids = [line.strip() for line in f]
    num_of_classes = len(wnids)
    print(f'Number of wnids in Tiny-Imagenet: {num_of_classes}')

    # Get all the wnids and trim xx% of them (tracking all wnids left in and out)
    random.shuffle(wnids) # Randomize ordering of wnids

    ind_wnids = set(wnids[:int(num_of_classes * args.ind_class_ratio)])
    ood_wnids = set(wnids[int(num_of_classes * args.ind_class_ratio):])

    print(f"Number of IND wnids: {len(ind_wnids)}")
    print(f"Number of OOD wnids: {len(ood_wnids)}")

    # Loop over {wnids}/images and trim xx% of the examples (tracking all examples left in and out)
    for wnid in wnids:
        # Retrieve paths for specific wnid in both training and validation sets
        wnid_train_path = os.path.join(train_dataset_path, wnid) # ~/data/tiny-imagenet-200/tiny-imagenet-200/train/{wnid}
        train_images_path = os.path.join(wnid_train_path, 'images') # ~/data/tiny-imagenet-200/tiny-imagenet-200/train/{wnid}/images

        val_images_path = os.path.join(val_dataset_path, wnid) # ~/data/tiny-imagenet-200/tiny-imagenet-200/val_organized/{wnid}

        if wnid in ind_wnids:
            # ============ TRAINING Dataset Processing============ #
            # Retrieve a list of all wnid specific JPEG file names
            wnid_image_file_names = []
            for filename in os.listdir(train_images_path):
                image_path = os.path.join(train_images_path, filename)
                if os.path.isfile(image_path):
                    wnid_image_file_names.append(filename)

            random.shuffle(wnid_image_file_names) # Shuffle images names
            left_in_image_names = set(wnid_image_file_names[:int(args.ind_example_ratio * len(wnid_image_file_names))])
            # left_our_image_names = set(wnid_image_file_names[:int(args.ind_example_ratio * len(wnid_image_file_names))])
            for filename in os.listdir(train_images_path):
                image_path = os.path.join(train_images_path, filename)
                if filename in left_in_image_names:
                    try:
                        if os.path.isfile(image_path):
                            # Create wnid directory is needed
                            destination_dir_path = os.path.join(save_path_train_ind_left_in, wnid)
                            if not os.path.exists(destination_dir_path):
                                os.mkdir(destination_dir_path)

                            destination_dir_images_path = os.path.join(destination_dir_path, "images")
                            if not os.path.exists(destination_dir_images_path):
                                os.mkdir(destination_dir_images_path)

                            destination_file_path = os.path.join(destination_dir_images_path, filename)
                            shutil.copyfile(image_path, destination_file_path)

                    except Exception as e:
                        print(f"Error copying JPEG at location: '{image_path}': {e}")
                else:
                    try:
                        if os.path.isfile(image_path):
                            destination_dir_path = os.path.join(save_path_train_ind_left_out, wnid)
                            if not os.path.exists(destination_dir_path):
                                os.mkdir(destination_dir_path)

                            destination_dir_images_path = os.path.join(destination_dir_path, "images")
                            if not os.path.exists(destination_dir_images_path):
                                os.mkdir(destination_dir_images_path)
                            destination_file_path = os.path.join(destination_dir_images_path, filename)
                            shutil.copyfile(image_path, destination_file_path)
                    except Exception as e:
                        print(f"Error copying JPEG at location: '{image_path}': {e}")

            # ============ Validation Dataset Processing ============ #
            try:
                destination_dir_path = os.path.join(save_path_val_ind, wnid)
                shutil.copytree(val_images_path, destination_dir_path)
            except Exception as e:
                print(f"Error while copying over {wnid} validation set with error {e}")

        # OOD wnids
        else:
            try:
                destination_train_dir_path = os.path.join(save_path_train_ood, wnid)
                shutil.copytree(wnid_train_path, destination_train_dir_path)

                destination_val_dir_path = os.path.join(save_path_val_ood, wnid)
                shutil.copytree(val_images_path, destination_val_dir_path)

            except Exception as e:
                print(f"Error while copying over {wnid} training or validation set with error {e}")

    class_info = {
        'pretrain_classes': list(ind_wnids), # List of original class indices (0-199) for ID
        'left_out_classes': list(ood_wnids),   # List of original class indices (0-199) for OOD
        'num_of_classes': num_of_classes,
    }
    return save_path_train_ind_left_in, save_path_val_ind, class_info
