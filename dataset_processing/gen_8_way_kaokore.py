import pandas as pd
import csv
import os
import shutil

def load_labels(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        return [{
            headers[column_index]: row[column_index]
            for column_index in range(len(row))
        }
            for row in reader]


if __name__ == '__main__':
    root = '../datasets/kaokore'
    data_split =  ['train', 'dev', 'test']
    lbls =  ['gender', 'status']

    categs ={'gender': ['male', 'female'] , 'status': ['noble', 'warrior', 'incarnation', 'commoner']}
    for split in data_split:
        for gender in categs['gender']:
            for status in categs['status']:
                os.makedirs('../datasets/kaokore_imagenet_style_8way/'+split+'/'+gender+'-'+status, exist_ok=True)
    labels = load_labels(os.path.join(root, 'labels.csv'))
    print('Generating imagenet style dataset')
    for split in data_split:
        for gender in categs['gender']:
            for status in categs['status']:
                current_path = '../datasets/kaokore_imagenet_style_8way/'#+split+'/'+gender+'-'+status+'/'
                for label_entry in labels:
                    if os.path.exists(os.path.join(root, 'images_256', label_entry['image'])):
                        shutil.copy(os.path.join(root, 'images_256', label_entry['image']),
                                    os.path.join(current_path, 
                                                 label_entry['set'],
                                                 categs['gender'][int(label_entry['gender'])]+'-'+categs['status'][int(label_entry['status'])],
                                                 label_entry['image'])
                                    )
print('Finished generating samples')


