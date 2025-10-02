import os
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.utils.data import Dataset, DataLoader
import librosa
from scipy.signal import butter, lfilter
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
from prerprocess import bandpass_filter, normalize_audio, extract_instances_sliding_window, pad_instances, extract_features

# setting
CHECKPOINT_PATH = "/.../2icbhi_model.pth"   # your pretrain MIL model
TEST_DATA_DIR = "/.../data"             # Four folders: Both, Crackle, Normal, Wheeze
SAMPLE_RATE = 16000
WINDOW_SIZE = 2.0    
STEP_SIZE = 1.0      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5      # threshold for crackle/wheeze
DETAIL_TSV = "detail.tsv"
BATCH_SIZE = 4
# ------------------------------------------------------

# multi-hot mapping
labels = {
    'Crackle': [1, 0],
    'Normal' : [0, 0],
    'Wheeze' : [0, 1],
    'Both'   : [1, 1]
}
# single-label mapping for metric (order matters)
single_label_list = ['Crackle', 'Normal', 'Wheeze']  

def multi_to_single_label(multi_vec):
    # multi_vec: [crackle, wheeze]
    tup = (int(multi_vec[0]), int(multi_vec[1]))
    if tup == (0,0): return 'Normal'
    if tup == (1,0): return 'Crackle'
    if tup == (0,1): return 'Wheeze'
    if tup == (1,1): return 'Both'
    return 'Unknown'

class MILDataset(Dataset):
    def __init__(self, bags, labels, filenames, intervals):
        self.bags = bags
        self.labels = labels
        self.filenames = filenames
        self.intervals = intervals
    def __len__(self):
        return len(self.bags)
    def __getitem__(self, idx):
        bag = self.bags[idx]  
        bag_t = torch.tensor(bag, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        filename = self.filenames[idx]
        intervals = self.intervals[idx]
        return bag_t, label, filename, intervals

def mil_collate_fn(batch):
    bags_list, labels_list, filenames_list, intervals_list = [], [], [], []
    for bag, label, filename, intervals in batch:
        bags_list.append(bag)
        labels_list.append(label)
        filenames_list.append(filename)
        intervals_list.append(intervals)
    # pad instances
    max_instances = max([b.shape[0] for b in bags_list])
    max_audio_len = max([b.shape[1] for b in bags_list])
    padded_bags = []
    for b in bags_list:
        
        if b.shape[0] < max_instances:
            pad_inst = torch.zeros((max_instances - b.shape[0], b.shape[1]), dtype=torch.float32)
            b = torch.cat((b, pad_inst), dim=0)

        if b.shape[1] < max_audio_len:
            pad_len = torch.zeros((b.shape[0], max_audio_len - b.shape[1]), dtype=torch.float32)
            b = torch.cat((b, pad_len), dim=1)
        padded_bags.append(b)
    bags_batch = torch.stack(padded_bags, dim=0)
    labels_batch = torch.stack(labels_list, dim=0)
    return bags_batch, labels_batch, filenames_list, intervals_list

#  Data loading 
def try_load_model(checkpoint_path, device):
    

    loaded = torch.load(checkpoint_path, map_location=device)

    if isinstance(loaded, nn.Module):
        model = loaded.to(device)
        return model
    
    if isinstance(loaded, dict):
        if 'model' in loaded and isinstance(loaded['model'], dict):
            state = loaded['model']
        else:
            state = loaded

        try:
            import model_defs  
            if hasattr(model_defs, 'MILModelBinary'):
                model = model_defs.MILModelBinary()
                model.load_state_dict(state, strict=False)
                return model.to(device)

        except Exception as e:
            print(" check model_defs.py exists")
            print(" error:", e)

        raise RuntimeError("checkpoint:state_dict, can't find model")
    raise RuntimeError("wrong checkpoint")

# Load data
def load_data_with_sliding_window(data_dir, window_size=2.0, step_size=1.0):
    bags, bag_labels, filenames, instance_intervals = [], [], [], []
    for label_name, label_vec in labels.items():
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if not fname.lower().endswith('.wav'):
                continue
            path = os.path.join(label_dir, fname)
            try:
                y, sr = librosa.load(path, sr=SAMPLE_RATE)
                if y is None or len(y) == 0:
                    print(f"pass: {path}")
                    continue
                y = bandpass_filter(y, sr)
                y = normalize_audio(y)
                instances = extract_instances_sliding_window(y, sr, window_size, step_size)
                if not instances:
                    continue
                padded_instances = pad_instances(instances)
                if not padded_instances:
                    continue
                feats = extract_features(padded_instances, sr) 
                if feats.size == 0:
                    continue
                intervals_for_bag = [(onset, offset) for _, onset, offset in padded_instances]
                bags.append(feats)
                bag_labels.append(label_vec)
                filenames.append(fname)
                instance_intervals.append(intervals_for_bag)
            except Exception as e:
                print(f"process {path} error：{e}")
    return bags, bag_labels, filenames, instance_intervals

# forward
def run_inference(checkpoint_path, test_data_dir):
    print(f"Device: {DEVICE}")
    print("loading data...")
    bags, bag_labels, filenames, instance_intervals = load_data_with_sliding_window(test_data_dir, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    if len(bags) == 0:
        raise RuntimeError("Check TEST_DATA_DIR")
    dataset = MILDataset(bags, bag_labels, filenames, instance_intervals)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=mil_collate_fn)
    # load model
    print("loading model...")
    model = try_load_model(checkpoint_path, DEVICE)
    model.eval()
    # outputs collectors
    all_y_true_multi = []
    all_y_pred_multi = []
    all_y_true_single = []
    all_y_pred_single = []
    # open detail.tsv
    with open(DETAIL_TSV, 'w', newline='') as fdet:
        writer = csv.writer(fdet, delimiter='\t')
        writer.writerow(['filename', 'onset', 'offset', 'Crackle_score', 'Wheeze_score', 'event_label'])
        with torch.no_grad():
            for bags_batch, labels_batch, filenames_batch, intervals_batch in loader:
                
                B, N, L = bags_batch.shape
                bags_batch = bags_batch.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)
                out = model(bags_batch, mixup_lambda=None, return_bag_embedding=False, return_attention=False)
                if isinstance(out, tuple) and len(out) >= 2:
                    bag_outputs = out[0]
                    instance_outputs = out[1] 
                else:
                    raise RuntimeError("model return error")
                probs = torch.sigmoid(bag_outputs)  
                predicted_multi = (probs >= THRESHOLD).long().cpu().numpy()  
               
                inst_crackle = torch.sigmoid(instance_outputs[0]).cpu().numpy()  
                inst_wheeze  = torch.sigmoid(instance_outputs[1]).cpu().numpy()  
                true_multi = labels_batch.cpu().numpy().astype(int)
                for i in range(B):
                    true_vec = true_multi[i]
                    pred_vec = predicted_multi[i]
                    all_y_true_multi.append(true_vec)
                    all_y_pred_multi.append(pred_vec)
                    true_label_name = multi_to_single_label(true_vec)
                    pred_label_name = multi_to_single_label(pred_vec)
                    #  detail.tsv 
                    fname = filenames_batch[i]
                    intervals = intervals_batch[i]
                    num_inst = len(intervals)
                    for j in range(num_inst):
                        cscore = float(inst_crackle[i, j, 0]) if inst_crackle.shape[2] >= 1 else float(inst_crackle[i, j])
                        wscore = float(inst_wheeze[i, j, 0]) if inst_wheeze.shape[2] >= 1 else float(inst_wheeze[i, j])
                        writer.writerow([fname, intervals[j][0], intervals[j][1], f"{cscore:.6f}", f"{wscore:.6f}", pred_label_name])

if __name__ == "__main__":
    print("Start Processing...")
    run_inference(CHECKPOINT_PATH, TEST_DATA_DIR)
    print("Finish")
