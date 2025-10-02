
# 🫁 CCIM-MIL: Inference for Respiratory Sound Classification using MIL

This repository provides a Multiple Instance Learning (MIL) inference pipeline for respiratory sound analysis. The model classifies respiratory events (e.g., Crackle, Wheeze, Normal) from auscultation recordings using weak labels, and provides localized prediction plots based on model attention.

---

## 🧠 Features
- Inference-only pipeline for MIL-based respiratory sound classification
- Supports multi-label prediction: **Crackle**, **Wheeze**, **Normal**
- Visualizes prediction results along timeline
- Uses pre-trained MIL model (download link below)
- Input format: `.wav` auscultation audio + `.tsv` with weak/strong labels

---

## 📁 Folder Structure
```
CCMIL/
├── CCMIL_inference_ICBHI.py    # Main inference script
├── model_defs.py               # CNN-based MIL model definition
├── prerprocess.py              # Preprocessing audio to mel features
├── plot.py                     # Visualization of predictions
├── data/                       # Example input audio files
├── tsv/
│   ├── ground_truth.tsv        # Ground-truth event intervals
│   └── detail.tsv              # Inference output intervals
├── output_plots_withtruth/    # Visualized prediction results
```

---

## 🔗 Pretrained Model

You can download the pretrained MIL model from Google Drive:

📥 **[Download MIL model (Google Drive)](https://drive.google.com/file/d/1GERG9U92WasK0xY6lkDF3o3jEjZjrBgc/view?usp=sharing)**

After downloading, place the model checkpoint (e.g., `best_model.pth`) in the working directory.

---

## 🧪 Example Inference

```bash
python CCMIL_inference_ICBHI.py
```

You can customize the input `.wav` file, output plot directory, or prediction threshold by modifying the script.

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- librosa
- numpy
- matplotlib
- pandas
- scipy

Install requirements (optional):

```bash
pip install -r requirements.txt
```

(*`requirements.txt`*)

---

## 📊 Output

- **TSV Output** (`tsv/detail.tsv`): predicted events and intervals
- **Prediction Plots** (`output_plots_withtruth/`): timeline barplots of predictions
    - Top: MIL prediction (Crackle / Wheeze)
    - Bottom: Ground truth (if available)

---

## 📸 Sample Output

The figure below shows the timeline of model-predicted Crackle and Wheeze events compared with the ground-truth annotations:

```
output_plots_withtruth/173_1b1_Al_sc_Meditron_1_2class.png
```

---

## 👤 Author

Developed by [Winson Yen](https://github.com/WinsonYen)

---

## 📄 License

MIT License

