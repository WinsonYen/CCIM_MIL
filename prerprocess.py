import os
import numpy as np
import torch
import torch.nn as nn
import csv
from torch.utils.data import Dataset, DataLoader
import librosa
from scipy.signal import butter, lfilter
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix

def bandpass_filter(y, sr, lowcut=80, highcut=2000):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(1, [low, high], btype='band')
    return lfilter(b, a, y)

def normalize_audio(y):
    denom = np.max(np.abs(y)) + 1e-9
    return y / denom

def extract_instances_sliding_window(y, sr, window_size, step_size):
    instances = []
    total_length = len(y)
    window_length = int(window_size * sr)
    step_length = int(step_size * sr)
    if total_length < window_length:
        onset = 0.0
        offset = total_length / sr
        instances.append((y, onset, offset))
        return instances
    for start in range(0, total_length - window_length + 1, step_length):
        end = start + window_length
        instance = y[start:end]
        onset = start / sr
        offset = end / sr
        instances.append((instance, onset, offset))
    return instances

def pad_instances(instances):
    
    if not instances:
        return []
    max_length = max(len(inst[0]) for inst in instances)
    padded = []
    for inst, onset, offset in instances:
        if len(inst) < max_length:
            repeat_factor = max_length // len(inst) + 1
            inst_rep = np.tile(inst, repeat_factor)[:max_length]
            padded.append((inst_rep.astype(np.float32), onset, offset))
        else:
            padded.append((inst[:max_length].astype(np.float32), onset, offset))
    return padded

def extract_features(instances, sr):
    
    feats = []
    for inst, _, _ in instances:
        feats.append(inst.astype(np.float32))
    if feats:
        return np.stack(feats)  
    else:
        return np.empty((0, 0), dtype=np.float32)

