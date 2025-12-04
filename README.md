# 3D Heart Mesh Deformation with Swin Transformer + GAT

Direct 3D mesh deformation for heart chamber reconstruction from CT and MRI scans using a **3D Swin Transformer encoder** and **Graph Attention Network (GAT)** for per-vertex displacement prediction.

## 📋 Overview

This project implements a **direct mesh deformation pipeline** for automatic heart mesh generation from medical imaging data (CT and MRI). The model learns to deform a smooth template mesh into the anatomically correct heart surface shape.

### Heart Structures Segmented:
- **Class 0**: Background
- **Class 1**: Left Ventricle (LV)
- **Class 2**: Right Ventricle (RV)
- **Class 3**: Left Atrium (LA)
- **Class 4**: Right Atrium (RA)
- **Class 5**: Myocardium
- **Class 6**: Aorta
- **Class 7**: Pulmonary Artery (PA)

## 🏗️ Model Architecture

### Swin Transformer Encoder + GAT Decoder
The model combines a **3D Swin Transformer** encoder with a **Graph Attention Network (GAT)** for direct mesh deformation:

```
Input: CT/MR Image
       ↓
    [Swin Transformer Encoder]
    - Hierarchical feature extraction
    - Window-based self-attention
    - Multi-scale features (4 levels)
       ↓
    [Feature Sampling]
    - Sample deep features at mesh vertices
    - L2 normalization (optional)
       ↓
    [Graph Attention Network (GAT)]
    - Per-vertex feature input
    - Edge-based attention mechanism
    - Predict vertex displacements
       ↓
Output: Deformed Mesh Vertices
```

**Key Features:**
- **Swin Transformer**: Global receptive field, hierarchical features
- **Graph Attention**: Learns mesh structure while predicting displacements
- **Edge-based Attention**: Memory efficient (avoids dense NxN)
- **Direct Mesh Deformation**: No marching cubes needed at inference
- **Chamfer Loss**: Ensures mesh accuracy to ground-truth surface

## 📊 How the Model Works

### 1. **Data Preparation** (`dataset.py`)
```python
from dataset import HeartSegmentationDataset

# Loads NIfTI files (.nii format) from archive/
# Resizes volumes to (160, 160, 80) for consistent input
# Normalizes intensity values
# Pairs images with segmentation labels (for training only)
```

**Dataset Organization:**
- Training images/labels are in `archive/ct_train/` and `archive/mr_train/`
- Test images are in `archive/ct_test/` and `archive/mr_test/`
- Each training sample has:
  - `*_image.nii`: Input CT/MR scan
  - `*_label.nii`: Ground-truth segmentation (used only for training)

### 2. **Model Architecture** (`model_transformer.py`)
```python
from model_transformer import get_transformer_model, MeshDeformationGAT

model = get_transformer_model(in_channels=1, num_classes=8, embed_dim=96,
                              depths=[2,2,6,2], num_heads=[3,6,12,24])
```

**Components:**
- **Swin Transformer Encoder**: Extracts hierarchical image features
- **Feature Sampling**: Samples deep features at mesh vertex locations
- **Graph Attention Network (GAT)**: Predicts per-vertex displacements
- **Frozen Encoder**: Only GAT is trained for stability

**Why Swin + GAT?**
- ✅ Swin captures global image context efficiently
- ✅ GAT respects mesh topology while predicting displacements
- ✅ Direct mesh deformation (no intermediate voxel predictions)
- ✅ Efficient: Feature sampling avoids dense operations

### 3. **Training Process** (`train_gat_deform.py`)

```bash
# Train the GAT mesh deformation head
python train_gat_deform.py --epochs 100 --lr 1e-4 --w_lap 0.015 --w_edge 0.008
```

**Loss Functions:**
```
Total Loss = w_ch × Chamfer + w_lap × Laplacian + w_edge × Edge
```

- **Chamfer Distance** (w_ch=1.0): Measures mesh-to-surface accuracy
- **Laplacian Smoothness** (w_lap=0.015): Encourages smooth meshes
- **Edge Length Regularization** (w_edge=0.008): Maintains mesh regularity

**Training Process:**
1. Load image and ground-truth label (from training data)
2. Extract surface mesh from label using marching cubes
3. Create smooth template sphere mesh
4. Sample Swin encoder features at mesh vertices
5. GAT predicts vertex displacements
6. Compute Chamfer + regularization losses
7. Backpropagate and update GAT weights
8. Log metrics to `gat_deform_logs.csv`

**Target Metrics:**
- Chamfer distance < 0.1 (ideally < 0.01)
- Weighted lap loss ~0.5, edge loss ~0.5 (when ch ~5.0)

## 🚀 Installation & Setup

### Step 1: Install Dependencies
```bash
# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt
```

**Or manually:**
```bash
pip install numpy torch nibabel scipy scikit-image pyvista einops
```

### Step 2: Prepare Your Dataset

#### Download/Organize Your Data
Create the following directory structure under `archive/`:
```
archive/
├── ct_train/          ← Your CT training images and labels
├── ct_test/           ← Your CT test images (labels optional)
├── mr_train/          ← Your MR training images and labels
└── mr_test/           ← Your MR test images (labels optional)
```

#### File Naming Convention
- **Training images:** `{modality}_train_{ID}_image.nii` (e.g., `ct_train_1001_image.nii`)
- **Training labels:** `{modality}_train_{ID}_label.nii` (e.g., `ct_train_1001_label.nii`)
- **Test images:** `{modality}_test_{ID}_image.nii` (e.g., `ct_test_2001_image.nii`)

#### Label Format
Labels must be integer-valued NIfTI files with voxel values:
```
0: Background
1: Left Ventricle (LV)
2: Right Ventricle (RV)
3: Left Atrium (LA)
4: Right Atrium (RA)
5: Myocardium
6: Aorta
7: Pulmonary Artery (PA)
```

**Where to Add Your Dataset:**
```
1. Create archive/ directory in project root
2. Create subdirectories: ct_train, ct_test, mr_train, mr_test
3. Copy your .nii files into the appropriate directories
4. Naming must match: {modality}_{split}_{ID}_{type}.nii
```

## 📁 Dataset Structure

Your data should be organized as follows:
```
archive/
├── ct_train/
│   ├── ct_train_1001_image.nii
│   ├── ct_train_1001_label.nii
│   ├── ct_train_1002_image.nii
│   ├── ct_train_1002_label.nii
│   └── ... (more training samples)
├── ct_test/
│   ├── ct_test_2001_image.nii
│   ├── ct_test_2002_image.nii
│   └── ... (test samples, labels optional)
├── mr_train/
│   ├── mr_train_1001_image.nii
│   ├── mr_train_1001_label.nii
│   └── ...
└── mr_test/
    ├── mr_test_2001_image.nii
    └── ...
```

**Required for Training:** Images + Labels in `ct_train/` and/or `mr_train/`  
**Required for Inference:** Only images in `ct_test/`, `mr_test/`, or any custom path

## 🎯 Usage

### Step 1: Train the Swin Encoder (Optional)
If you already have `checkpoints_transformer/best_model.pth`, skip this.

Otherwise, train the Swin Transformer encoder on segmentation:
```bash
python train.py --epochs 100 --batch_size 2 --device cuda
```

This produces: `checkpoints_transformer/best_model.pth`

### Step 2: Train the GAT Mesh Deformation Head

**Single-template deformation (single unified mesh for all structures):**
```bash
python train_gat_deform.py `
  --epochs 100 `
  --lr 1e-4 `
  --w_lap 0.015 `
  --w_edge 0.008 `
  --reg_warmup_epochs 0 `
  --use_ct `
  --use_mr
```

This produces checkpoints in `checkpoints_transformer/` and logs in `gat_deform_logs.csv`.

**Multi-structure deformation (separate meshes per heart structure):**
```bash
python train_gat_multi_deform.py `
  --epochs 100 `
  --lr 1e-4 `
  --w_lap 0.015 `
  --w_edge 0.008 `
  --classes 1 2 3 4 5 6 7
```

### Step 3: Run Inference

**Single-template mesh deformation:**
```bash
python deform_reconstruct.py `
  --input ./archive/ct_test/ct_test_2001_image.nii `
  --encoder_ckpt ./checkpoints_transformer/best_model.pth `
  --gat_ckpt ./checkpoints_transformer/gat_deform_epoch_100.pth `
  --output_dir ./predictions_transformer `
  --show_template `
  --show_deformed
```

**Multi-structure mesh deformation:**
```bash
python deform_multi_reconstruct.py `
  --input ./archive/ct_test/ct_test_2001_image.nii `
  --encoder_ckpt ./checkpoints_transformer/best_model.pth `
  --gat_ckpt ./checkpoints_transformer/gat_multi_deform_epoch_100.pth `
  --output_dir ./predictions_transformer
```

**Full segmentation + visualization (voxel-wise):**
```bash
python reconstruct_heart.py `
  --input ./archive/ct_test/ct_test_2001_image.nii `
  --checkpoint ./checkpoints_transformer/best_model.pth `
  --device auto
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Initial learning rate |
| `--w_lap` | 0.015 | Laplacian smoothness weight |
| `--w_edge` | 0.008 | Edge length regularization weight |
| `--w_ch` | 1.0 | Chamfer distance weight (main loss) |
| `--reg_warmup_epochs` | 0 | Warmup regularization from 0 to target |
| `--feat_levels` | deep | Feature extraction: 'deep' or 'multi' |
| `--feat_norm` | ln | Feature normalization: 'none', 'l2', or 'ln' |
| `--hidden` | 256 | GAT hidden dimension |
| `--heads` | 4 | GAT attention heads |
| `--layers` | 3 | GAT depth layers |
| `--grad_clip` | 1.0 | Gradient clipping value |

## 📈 Expected Performance

### Training Metrics (from `gat_deform_logs.csv`)
```
Epoch    Loss     Chamfer   Laplacian  Edge      Offset
1        14.81    14.81     0.21       4.68      2.50
10       8.42     8.30      0.15       3.22      1.85
50       2.15     2.08      0.04       1.02      0.95
100      0.85     0.80      0.03       0.78      0.65
```

**Target Goals:**
- **Chamfer Distance** < 0.1 (or < 0.01 for high accuracy)
- **Weighted Laplacian** ~0.5 (ensures smoothness)
- **Weighted Edge** ~0.5 (ensures regularity)
- **Mean Displacement** stabilizes over epochs

### Factors Affecting Convergence
- Data quality and label consistency
- Number of training samples
- Regularization weights (w_lap, w_edge)
- Learning rate and scheduler
- Mesh resolution (template vertices)

## 🔧 Customization

### Adjust Model Capacity
```bash
# Increase GAT hidden dimension for more capacity
python train_gat_deform.py --hidden 512 --heads 8 --layers 4

# Use multi-level features from encoder
python train_gat_deform.py --feat_levels multi
```

### Change Input Size
```bash
# Higher resolution (requires more GPU memory)
python train_gat_deform.py --target_shape 200 200 100
```

### Tune Regularization
```bash
# Strong smoothness and edge constraints
python train_gat_deform.py --w_lap 0.05 --w_edge 0.02

# Weak regularization (allow more deformation)
python train_gat_deform.py --w_lap 0.005 --w_edge 0.002
```

### Learning Rate Schedule
```bash
# Use step-based scheduler instead of cosine
python train_gat_deform.py --lr_scheduler step --step_size 20 --gamma 0.5

# Use cosine annealing (default)
python train_gat_deform.py --lr_scheduler cosine --min_lr 1e-6
```

## 🧠 Understanding the Pipeline

### **Complete Training Workflow**
```
1. Load Image (CT/MR)
   ↓
2. Extract Features (Swin Encoder)
   ↓
3. Extract Ground-truth Surface (Marching Cubes on Label)
   ↓
4. Create Template Sphere Mesh
   ↓
5. Sample Features at Mesh Vertices
   ↓
6. GAT Predicts Vertex Displacements
   ↓
7. Compute Loss (Chamfer + Laplacian + Edge)
   ↓
8. Backpropagate and Update GAT
```

### **Complete Inference Workflow**
```
1. Load Image (CT/MR) — NO LABEL NEEDED
   ↓
2. Extract Features (Swin Encoder)
   ↓
3. Estimate Template Center/Radius (from features or optional seg)
   ↓
4. Create Template Sphere or Ellipsoid
   ↓
5. Sample Features at Mesh Vertices
   ↓
6. Load Trained GAT
   ↓
7. GAT Predicts Vertex Displacements
   ↓
8. Output: Deformed Mesh (VTP file)
   ↓
9. Optional: Visualize and Save
```

### **Key Differences from Voxel-wise Segmentation**
| Aspect | Voxel-wise (train.py) | Mesh Deformation (train_gat_deform.py) |
|--------|----------------------|----------------------------------------|
| **Training** | Requires label | Requires label (ground-truth surface) |
| **Inference** | Requires segmentation | Direct mesh output, no label needed |
| **Output** | Probability maps | Mesh vertices + faces (.vtp) |
| **Accuracy** | Per-voxel Dice | Chamfer distance to surface |
| **Speed** | Slow (whole volume) | Fast (template + displacement only) |

## 📚 Project Files

| File | Purpose |
|------|---------|
| `dataset.py` | Data loading, preprocessing, augmentation for CT/MR |
| `model_transformer.py` | Swin Transformer encoder, GAT decoder, full model |
| `mesh_losses.py` | Chamfer, Laplacian, edge loss functions |
| `train_gat_deform.py` | Training script for single-template mesh deformation |
| `train_gat_multi_deform.py` | Training script for per-structure (7 classes) mesh deformation |
| `deform_reconstruct.py` | Inference script for single-template mesh output |
| `deform_multi_reconstruct.py` | Inference script for multi-structure mesh output |
| `reconstruct_heart.py` | Full voxel-wise segmentation + optional mesh visualization |
| `train.py` | (Optional) Legacy voxel-wise segmentation trainer |
| `requirements.txt` | Python package dependencies |

## 🎓 Key Concepts

### **Chamfer Distance**
```
CD = (1/|P₁|)Σ min ||p - q|| + (1/|P₂|)Σ min ||q - p||
          p∈P₁  q∈P₂                  q∈P₂  p∈P₁
```
- Measures average closest-point distance between two point clouds
- Lower values indicate better surface alignment
- Main metric for mesh deformation accuracy

### **Laplacian Smoothness Loss**
```
L_lap = Σ ||vᵢ - mean(neighbors(vᵢ))||²
```
- Encourages vertices to stay close to neighbors
- Prevents crumpled, noisy meshes
- Essential for anatomically plausible shapes

### **Edge Length Loss**
```
L_edge = Σ ||eᵢ|| - target_length||²
```
- Penalizes very long or short edges
- Maintains uniform mesh quality
- Prevents mesh degeneration

### **Why Direct Mesh Deformation?**
- ✅ **Explicit geometry:** Direct mesh output, not voxels
- ✅ **Memory efficient:** Process only surface, not entire volume
- ✅ **Fast inference:** No marching cubes at test time
- ✅ **Differentiable:** End-to-end gradient flow
- ✅ **Smooth output:** Regularization enforces plausibility

## 🐛 Troubleshooting

### Out of Memory (CUDA)
```bash
# Reduce input size
python train_gat_deform.py --target_shape 128 128 64

# Use smaller GAT
python train_gat_deform.py --hidden 128 --heads 2 --layers 2

# Use single-level features
python train_gat_deform.py --feat_levels deep
```

### Poor Mesh Quality (Crumpled, Distorted)
```bash
# Increase regularization
python train_gat_deform.py --w_lap 0.05 --w_edge 0.02

# Train longer
python train_gat_deform.py --epochs 200

# Reduce learning rate
python train_gat_deform.py --lr 5e-5
```

### Chamfer Distance Not Decreasing
```bash
# Check feature diagnostics
python train_gat_deform.py --log_feature_detail

# Increase GAT capacity
python train_gat_deform.py --hidden 512 --layers 4

# Verify data format (must be NIfTI with correct labels)
```

### Model Not Learning
- Verify dataset structure (check `archive/` folders)
- Check label values (0-7, no random labels)
- Ensure encoder checkpoint exists: `checkpoints_transformer/best_model.pth`
- Review training logs for NaN or extremely large losses

## 📚 References

- **Swin Transformer**: Ze Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
- **Graph Attention Networks**: Petar Veličković et al., "Graph Attention Networks", ICLR 2018
- **Chamfer Distance**: Fan et al., "A Point Set Generation Network for 3D Object Reconstruction from a Single Image", CVPR 2017
- **3D U-Net**: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation", MICCAI 2016
- **Medical Image Segmentation**: Survey of deep learning methods for 3D volumetric segmentation

## 🆘 Getting Help

### Common Issues & Solutions
1. **"ModuleNotFoundError: No module named 'nibabel'"**  
   → Run: `pip install nibabel`

2. **"CUDA out of memory"**  
   → Reduce target_shape or hidden dimension

3. **"FileNotFoundError: archive/"**  
   → Create `archive/` folder with correct structure (see Dataset Structure section)

4. **"No training data found"**  
   → Check file naming: must be `{modality}_{split}_{ID}_{type}.nii`

5. **"Chamfer loss is NaN"**  
   → May indicate no foreground voxels in label; check label quality

### Performance Tips
- Use `--log_feature_detail` flag to diagnose feature sampling issues
- Check `gat_deform_logs.csv` for loss trends
- Start with fewer epochs (10-20) for quick testing
- Monitor GPU memory with nvidia-smi (for CUDA)

---

**Happy Heart Meshing! 🫀✨**

---

## Quick Start (TL;DR)

```bash
# 1. Organize data in archive/
# 2. Train encoder (if no best_model.pth)
python train.py --epochs 50

# 3. Train mesh deformation
python train_gat_deform.py --epochs 100 --lr 1e-4

# 4. Run inference
python deform_reconstruct.py --input archive/ct_test/ct_test_2001_image.nii

# 5. View results in predictions_transformer/
```
