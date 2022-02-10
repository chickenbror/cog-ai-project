"""
Parses Flickr caption file -> (image_tensor, ANP_multi_hot_tensor, adj_multi_hot_tensor, noun_multi_hot_tensor)
"""

import torch
from torchvision import transforms # Preprocess jpg images
from PIL import Image
import spacy
from tqdm import tqdm
import pickle

nlp = spacy.load("en_core_web_sm")
def get_ANP(caption, add_empty_adj=False):
    '''Parses a text and returns a list of adj-noun pairs'''
    doc = nlp(caption)
    ANPs = []
    for i,token in enumerate(doc):
        
        # Look for a noun
        if token.pos_ not in ('NOUN'):
            continue

        noun = token.lemma_.lower() # Noun in lemma form
        try:
            # Look for preceding adj
            if doc[i-1].pos_ == 'ADJ':
                adj = doc[i-1].lemma_.lower() # Adj in lemma form
                ANPs.append((adj, noun))
            else:
                if add_empty_adj:
                    ANPs.append(('-', noun))
        except IndexError:
            pass
    return ANPs


def get_imgs_with_ANPs(fn_caption_file, min_freq=150, keep_empty_adj=False,
                       pickle_fn="saved_fn_anp_dict.pkl"):
    '''Iterate through the img_fn,caption and return img_fn:[(adj,noun)...] '''
    
    # Read previously saved pickle file) => get fn:[ANP...]
    try:
        with open(pickle_fn, "rb") as f:
            print("Reading previously saved dictionary of Filename:Adj-Noun-Pairs...")
            fn_ANP_dict = pickle.load(f)
            
    except FileNotFoundError:
        # Read the txt file => get fn:[ANP...]
        print("Reading captions and looking for Adj-Noun Pairs...")
        with open(fn_caption_file) as f:
            fn_ANP_dict = {}
            lines = [l for l in f]
            for line in tqdm(lines):
                if len(line.split(','))!=2:
                    continue
                fn, cap = line.strip('\n').split(',')
                if fn=='image':
                    continue
                ANPs = get_ANP(cap, add_empty_adj=True)
                if ANPs == []:
                    continue
                # add new key, or expend the ANP list    
                if fn not in fn_ANP_dict:
                    fn_ANP_dict[fn] = ANPs
                else:
                    current_ANPs = fn_ANP_dict[fn]
                    new_ANPs = [x for x in ANPs if x not in current_ANPs]
                    fn_ANP_dict[fn].extend(new_ANPs)
                
        # Save the parsed fn-ANPs dict for future use   
        with open(pickle_fn, "wb") as f:
            pickle.dump(fn_ANP_dict, f)
    
    #===================================================
    #Count the frequency of adjs and nouns
    def count_adj_noun(fn_ANP_dict):
        a_counts, n_counts = {}, {}

        for anp_list in fn_ANP_dict.values():
            for (a,n) in anp_list:
                a_counts[a] = a_counts.get(a,0)+1
                n_counts[n] = n_counts.get(n,0)+1
        return a_counts, n_counts
    
    a_counts, n_counts = count_adj_noun(fn_ANP_dict)  
    if not keep_empty_adj and '-' in a_counts:
        a_counts.pop('-')
    print(f"Before filtering: {len(fn_ANP_dict)} images with ANPs.")
    print(f"{len(a_counts)} adjs, {len(n_counts)} nouns.")
    
    #===================================================
    #Filter adjs and nouns that have appeared minimum N times
    def get_frequent_words(word_count_dict):
        return [w for w,freq in word_count_dict.items() if freq>=min_freq]
    
    print(f"\nAfter filtering: Only keep the adjs or nouns that are annotated in at least {min_freq} images:")
    frequent_adjs = get_frequent_words(a_counts)
    frequent_nouns = get_frequent_words(n_counts)
    ANP_combos = [(a,n) for a in frequent_adjs for n in frequent_nouns]
    print(f"{len(frequent_adjs)} freq adjs X {len(frequent_nouns)} freq nouns = {len(ANP_combos)} possible combinations")
    
    #===================================================
    #Filter images that contain the frequent adjs/nouns
    filtered_fn_ANPs = {}
    for fn, ANPs in fn_ANP_dict.items():
        frequent_ANPs = [(a,n) for a,n in ANPs if a in frequent_adjs and n in frequent_nouns]
        if frequent_ANPs:
            filtered_fn_ANPs[fn]=frequent_ANPs
            
    ANP_set=set()
    for anps in list(filtered_fn_ANPs.values()):
        ANP_set.update(anps)
        
    print(f"{len(filtered_fn_ANPs)} images with {len(ANP_set)} ANPs containing frequent adjs and nouns.")
    
    return frequent_adjs, frequent_nouns, ANP_combos, filtered_fn_ANPs


def file_to_tensor(img_path):
    # (PIL.Image obj) => pt tensor 3x224x224
    # load jpg directly as tensor of CxHxW to avoid needing converting from numpy and permuting

    preprocess_fn = transforms.Compose([
        transforms.Resize( 240 ), # img gets cropped if it isn't squared-shaped
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    # PIL Img obj
    with Image.open(img_path) as img:
        img_tensor = preprocess_fn(img)
        return img_tensor

def augment_img_tensors(batch_tensors):
#     tensor_to_PIL = transforms.ToPILImage()
    augment_fn = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                            shear=None, interpolation=transforms.InterpolationMode.NEAREST,
                                            fill=(255, 255, 255)),
                    transforms.ToTensor(),
                    ])
    # tensors->PIL objs-> randomly change angles, flip, etc ->new tensors
    new_tensors = [augment_fn(x) for x in batch_tensors]
    new_tensors = torch.stack(new_tensors)
    return new_tensors

def multilabels_to_tensor(list_labels, list_classes):
    '''Returns a tensor of length = num_classes, which is filled with 0 or 1'''
    zeros = torch.zeros(len(list_classes))
    for label in list_labels:
        if label in list_classes:
            idx = list_classes.index(label)
            zeros[idx] = 1
    return zeros

def convert_fn_anp_to_xy(fn_anp, img_path, ANP_classes, adj_classes, noun_classes):
    '''Converst a list of (filename, [labels...]) to 
    a list of (img_tensor, multilabels_tensor) tuples'''
    xy=[] 
    for fn, anp_list in tqdm(fn_anp):
        img = file_to_tensor(img_path+fn)
        anp_labels = multilabels_to_tensor(anp_list, ANP_classes)
        
        adjs, nouns = zip(*anp_list)
        adj_labels = multilabels_to_tensor(adjs, adj_classes)
        noun_labels = multilabels_to_tensor(nouns, noun_classes)
        xy.append((img, anp_labels, adj_labels, noun_labels))
    return xy