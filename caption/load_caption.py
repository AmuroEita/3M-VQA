import csv

csv_file_paths = [
  'Cap-4o/brazil_caption_img_only.csv',
  'Cap-4o/israel_caption_img_only.csv',
  'Cap-4o/japan_caption_img_only.csv',
  'Cap-4o/spain_caption_img_only.csv'
]

all_captions = [] 

def load_caption():
    for csv_file_path in csv_file_paths:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            captions = []
            for row in reader:
                captions.append(row['caption'])  
            all_captions.append(captions)  
            print(captions)  
    return all_captions  