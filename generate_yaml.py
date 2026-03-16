import os
import yaml
import pickle, gzip
import numpy as np
from collections import OrderedDict

# filename1 = '/data1/guanren/CSL-Daily/csl_clean.train'
# with gzip.open(filename1, "rb") as f:
#         loaded_object = pickle.load(f)
#         print(type(loaded_object))
#         print(len(loaded_object))

# filename2 = '/data1/guanren/CSL-Daily/csl-daily.dev'
# with gzip.open(filename2, "rb") as f:
#         loaded_object2 = pickle.load(f)
#         print(type(loaded_object2))
#         print(len(loaded_object2))

# filename3 = '/data1/guanren/CSL-Daily/csl_clean.test'
# with gzip.open(filename3, "rb") as f:
#         loaded_object3 = pickle.load(f)
#         print(type(loaded_object3))
#         print(len(loaded_object3))

# text_dict = {}
# train_dataset = []
# for item in loaded_object:
#     text = item.get('text')
#     if text is not None:
#         if text not in text_dict:
#             train_dataset.append(item)
#             text_dict[text] = item

# print(len(train_dataset))

# text_dict2 = {}
# valid_dataset = []
# for item in loaded_object2:
#         text = item.get('text')
#         if text is not None:
#                 if text not in text_dict2:
#                         valid_dataset.append(item)
#                         text_dict2[text] = item

# print(len(valid_dataset))

# text_dict3 = {}
# test_dataset = []
# for item in loaded_object3:
#         text = item.get('text')
#         if text is not None:
#                 if text not in text_dict3:
#                         test_dataset.append(item)
#                         text_dict3[text] = item

# print(len(test_dataset))

root_directory = '/data1/guanren/American_poses'

entries = os.listdir(root_directory)
train_dataset_processed = []
valid_dataset_processed = []
test_dataset_processed = []

# for item in valid_dataset:
#         if item.get('name') in entries:
#                 valid_dataset_processed.append(item)

# for item in train_dataset:
#         if item.get('name') in entries:
#                 train_dataset_processed.append(item)

# for item in test_dataset:
#         if item.get('name') in entries:
#                 test_dataset_processed.append(item)

# print(len(valid_dataset_processed))
# print(len(train_dataset_processed))
# print(len(test_dataset_processed))


directory = '/data1/guanren/American_poses/source_body'

motions = {}
count=0
filtered_list = [item for item in entries if 'target_linker_hand' not in item]
filtered_list = [item for item in filtered_list if 'source_body' not in item]
filtered_list = [item for item in filtered_list if 'target_body' not in item]
# for item in test_dataset:
for item in filtered_list:

        file_path = os.path.join(directory, item)
        data = np.load(file_path+'.npy', allow_pickle=True)
        ordered_dict = OrderedDict(data.item())
        if ordered_dict['rotation']['arr'].shape[0]<10:
            continue
        else:
            if ordered_dict['rotation']['arr'].shape[0]>=200:
                count+=1
            # key = '\''+item.get('name')+'\''
                key = '\''+item+'\''
                motions[key] = {
                    'description': 'American Sign Language',
                    'difficulty': 4,
                    'trim_beg': -1,
                    'trim_end': -1,
                    'weight': 1.0
                }

    # if count==13000:
    #     break

yaml_content = {'motions': motions}
print(count)
output_file = '/home/guanren/expressive-humanoid/ASE/ase/poselib/data/configs/motions_American_sign_language_hard.yaml'

with open(output_file, 'w') as file:
    yaml.dump(yaml_content, file, default_flow_style=False, allow_unicode=True)
