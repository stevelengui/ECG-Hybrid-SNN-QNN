# 🫀 ECG Arrhythmia Classification - Hybrid SNN-QNN

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/stevelengui/ECG-Hybrid-SNN-QNN.git
cd ECG-Hybrid-SNN-QNN

# Install Python dependencies
pip install -r requirements.txt

# Train the model (automatically downloads MIT-BIH dataset)
python src/model.py

# Compile for RISC-V
make PLATFORM=hifive1  # For HiFive1 (RV32IMAC)
make PLATFORM=k210     # For Kendryte K210 (RV64GC)

📊 Results
Accuracy: 97.49% on MIT-BIH Arrhythmia Database

Model Size: 7.51 KB (under 9.1 KB target ✓)

Thermal Control: ΔT ≤ 5°C for medical applications

Precision Switching: 8-bit ↔ 4-bit ↔ 2-bit adaptive

Inference Latency: 8.2 ms (theoretical)

🏗️ Architecture
Hybrid architecture combining:

Temporal Path (SNN): LIF neurons + LSTM for rhythm analysis

Spatial Path (QNN): Quantized CNN for feature extraction

Dynamic Fusion: Attention-based feature router

Thermal Management: Predictive scheduler with precision switching

🔧 Hardware Targets
Platform	Architecture	Frequency	Binary Size
HiFive1	RV32IMAC	32 MHz	16.3 KB
Kendryte K210	RV64GC	400 MHz	29.0 KB
📝 Citation
If you use this work, please cite:

bibtex
@software{lengui_ecg_2025,
  author = {Lengui, Steve},
  title = {ECG Hybrid SNN-QNN for Edge Devices},
  url = {https://github.com/stevelengui/ECG-Hybrid-SNN-QNN},
  year = {2025}
}
📄 License
MIT License - See https://mit-license.org/ for details.
