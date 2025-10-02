import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_audio_file(audio_file, model_type='3', include_attention=False,
                    tsv_folder='.', output_folder='output_plots'):
    
    test_path   = os.path.join(tsv_folder, "/.../ground_truth.tsv")     # put your path here
    detail2_path = os.path.join(tsv_folder, "/.../detail.tsv")          # put your path here
    
    test_df   = pd.read_csv(test_path, sep="\t")
    detail2_df = pd.read_csv(detail2_path, sep="\t")   
    gt_data = test_df[test_df["filename"] == audio_file]
    

    if model_type == '2':
        pred_data = detail2_df[detail2_df["filename"] == audio_file]
        classes = ["Crackle", "Wheeze"]
        pred_colors = {"Crackle": "blue", "Wheeze": "red"}
    
    max_offset_pred = pred_data["offset"].max() if not pred_data.empty else 0
    max_offset_gt   = gt_data["offset"].max()   if not gt_data.empty   else 0
    max_offset_att  = 0
    max_time = max(max_offset_pred, max_offset_gt, max_offset_att)
    if np.isnan(max_time) or max_time == 0:
        max_time = max_offset_pred

    time_points = np.arange(0, max_time, 0.1)
    
    score_arrays = {cls: np.zeros_like(time_points) for cls in classes}
    counts = np.zeros_like(time_points)
    
    for _, row in pred_data.iterrows():
        start_idx = int(row["onset"] * 10)
        end_idx   = int(row["offset"] * 10)
        for cls in classes:
            col_name = f"{cls}_score"
            score_arrays[cls][start_idx:end_idx] += row[col_name]
        counts[start_idx:end_idx] += 1
        
    for cls in classes:
        score_arrays[cls] = np.divide(score_arrays[cls], counts,
                                       out=np.zeros_like(score_arrays[cls]),
                                       where=(counts > 0))
    
    # === Plotting ===
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    
    plt.figure(figsize=(16, 9))
    title_fs = 24
    axis_fs = 20
    
    for cls in classes:
        plt.plot(time_points, score_arrays[cls],
                 color=pred_colors[cls], label=f"Predicted {cls}", linewidth=4.0)    
    
    gt_y_mapping = {"normal": 1.1, "crackle": 1.2, "wheeze":1.3}
    gt_label_plotted = False
    for _, row in gt_data.iterrows():
        label_lower = str(row["event_label"]).lower()
        y_val = gt_y_mapping.get(label_lower, 1.3)
        if not gt_label_plotted:
            plt.hlines(y=y_val, xmin=row["onset"], xmax=row["offset"],
                       colors="black", linestyles="dashed", linewidth=4.0, label="Ground Truth")
            gt_label_plotted = True
        else:
            plt.hlines(y=y_val, xmin=row["onset"], xmax=row["offset"],
                       colors="black", linestyles="dashed", linewidth=4.0)
        mid_t = (row["onset"] + row["offset"]) / 2
        plt.text(mid_t, y_val + 0.02, label_lower,
                 ha="center", fontsize=16, fontweight='bold')  
    
 
    plt.xlabel("Time (s)", fontsize=axis_fs) 
    plt.ylabel("Score", fontsize=axis_fs)
    title_str = f"Audio File: {audio_file} | {model_type}-class Model"
    plt.title(title_str, fontsize=title_fs)
    
    plt.legend(fontsize=30, loc='upper left', frameon=True, fancybox=True, framealpha=1.0,
               edgecolor='gray', facecolor='white', shadow=False, borderpad=1, handlelength=1,
               title_fontsize=20, prop={'weight': 'bold'})  
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(audio_file)[0]
    model_str = "3class" if model_type == '3' else "2class"
    att_str = "_att" if include_attention else ""    
    png_path = os.path.join(output_folder, f"{base_name}_{model_str}{att_str}.png")
    
    
    plt.savefig(png_path, dpi=500, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"Save PNG: {png_path}")

if __name__ == '__main__':
    audio_file_to_plot = "name.wav"         # put your path here
    chosen_model = '2'
    plot_attention = False
    tsv_folder = "."
    output_folder = "output_plots_withtruth"
    
    plot_audio_file(audio_file_to_plot, model_type=chosen_model,
                    include_attention=plot_attention, tsv_folder=tsv_folder,
                    output_folder=output_folder)

