import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import mode

RAW_DIR = "./origin_data/PAMAP2_Dataset/Protocol"   # place original .dat files here
OUT_ROOT = "./data"       # output directory for processed .npy files
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
SEQ_LEN = 100     # sequence length (in samples)
STRIDE = 50       # sliding window stride (in samples)
for split in SPLITS.keys():
    os.makedirs(os.path.join(OUT_ROOT, split), exist_ok=True)


def load_subject(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None, comment='%', skip_blank_lines=True)
    df = df.dropna(how="all")
    return df
def window_data(x, y, seq_len=100, stride=50):
    X_seq, Y_seq = [], []
    for start in range(0, len(x) - seq_len + 1, stride):
        end = start + seq_len
        X_seq.append(x[start:end])
        # pick the mode of the labels in the window
        y_window = y[start:end]
        y_mode = mode(y_window, keepdims=False)[0]
        Y_seq.append(y_mode)
    return np.stack(X_seq), np.array(Y_seq)
def process_pamap2():
    imu_hand, imu_chest, imu_ankle, heart_rate, labels = [], [], [], [],[]

    for fname in sorted(os.listdir(RAW_DIR)):
        if fname.endswith(".dat"):
            path = os.path.join(RAW_DIR, fname)
            print(f"Processing {fname} ...")
            df = load_subject(path)
            df = df.dropna(subset=[1])  
            df = df[df.iloc[:, 1] != 0]  

            imu_hand.append(df.iloc[:, 3:20].values)
            imu_chest.append(df.iloc[:, 20:37].values)
            imu_ankle.append(df.iloc[:, 37:54].values)
            heart_rate.append(df.iloc[:, 2].values.reshape(-1, 1)) 
            labels.append(df.iloc[:, 1].values)

    imu_hand = np.vstack(imu_hand)
    imu_chest = np.vstack(imu_chest)
    imu_ankle = np.vstack(imu_ankle)
    heart_rate = np.vstack(heart_rate)
    labels = np.concatenate(labels)
    labels = labels.astype(int) - 1
    print("Label range:", labels.min().item(), labels.max().item())
    print(f"{len(labels)} frames in total")
    # windowing
    imu_hand, labels = window_data(imu_hand, labels, SEQ_LEN, STRIDE)
    imu_chest, _ = window_data(imu_chest, labels, SEQ_LEN, STRIDE)
    imu_ankle, _ = window_data(imu_ankle, labels, SEQ_LEN, STRIDE)
    heart_rate, _ = window_data(heart_rate, labels, SEQ_LEN, STRIDE)
    print(f"In sequence: imu_hand shape = {imu_hand.shape}")

    idx = np.arange(len(labels))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx, labels, test_size=1 - SPLITS["train"], stratify=labels, random_state=42
    )
    relative_val = SPLITS["val"] / (SPLITS["val"] + SPLITS["test"])
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp, test_size=1 - relative_val, stratify=y_tmp, random_state=42
    )
    splits = {"train": idx_train, "val": idx_val, "test": idx_test}

    def save_split(indices, split):
        out_dir = os.path.join(OUT_ROOT, split)
        np.save(os.path.join(out_dir, "imu_hand.npy"), imu_hand[indices])
        np.save(os.path.join(out_dir, "imu_chest.npy"), imu_chest[indices])
        np.save(os.path.join(out_dir, "imu_ankle.npy"), imu_ankle[indices])
        np.save(os.path.join(out_dir, "heart_rate.npy"), heart_rate[indices])
        np.save(os.path.join(out_dir, "labels.npy"), labels[indices])
        print(f"{split}: {len(indices)} sequences saved.")

    for split, idx_split in splits.items():
        save_split(idx_split, split)

if __name__ == "__main__":
    process_pamap2()
