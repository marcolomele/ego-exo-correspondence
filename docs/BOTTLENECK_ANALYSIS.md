# O-MaMa Training & Inference Bottleneck Analysis

## 1. DINO Feature Extraction (CRITICAL BOTTLENECK)

### Location
- `src/O-MaMa/descriptors/get_descriptors.py`: `extract_dense_features()` method
- Called in both training and inference for every batch

### Issues
1. **Heavy Model Forward Pass**
   - DINOv2 ViT-Base model runs on every batch
   - Even with `torch.no_grad()`, forward pass is computationally expensive
   - Processes full-resolution images (704x704 or 540x960 pixels)

2. **Memory Consumption**
   - Dense feature maps stored: `[B, C, H//patch_size, W//patch_size]`
   - For 704x704 image with patch_size=14: 50x50 = 2500 patches
   - For 540x960 image: 38x68 = 2584 patches
   - Feature dimension: 768
   - Memory per image: ~2500 × 768 × 4 bytes ≈ 8 MB per image
   - Batch size 12: ~92 MB just for dense features

3. **No Caching**
   - Features are recomputed every epoch
   - No pre-extraction or caching mechanism

### Impact
- **Training**: ~50-70% of total time likely spent on DINO forward passes
- **Inference**: Similar bottleneck, especially with batch_size=1

### Recommendations
- Pre-extract and cache DINO features for all images
- Use mixed precision (FP16) for DINO inference
- Consider using a smaller/faster backbone for initial experiments
- Batch DINO inference across multiple samples if possible

---

## 2. Data Loading Pipeline (HIGH PRIORITY)

### Location
- `src/O-MaMadataset/dataset_masks.py`: `__getitem__()` method
- `main.py`: DataLoader configuration

### Issues

1. **Low num_workers**
   ```python
   num_workers=1  # Line 187 in main.py
   ```
   - Only 1 worker process for data loading
   - CPU-GPU pipeline stalls while waiting for data

2. **Heavy I/O Operations Per Sample**
   - `cv2.imread()` for 2 images per sample
   - `np.load()` for masks (.npz) and bounding boxes (.npy)
   - Multiple file system accesses per sample
   - JSON loading for annotations

3. **CPU-Intensive Preprocessing**
   - Image resizing (cv2.resize) - done twice per sample
   - Delaunay triangulation for adjacency matrix (scipy)
   - Mask decoding (pycocotools)
   - Multiple tensor conversions

4. **Sequential Processing**
   - All preprocessing happens in `__getitem__()`
   - No parallelization of image loading

### Impact
- **Training**: GPU utilization likely <50% due to data loading stalls
- **Inference**: Less critical but still noticeable

### Recommendations
- Increase `num_workers` to 4-8 (or CPU count - 1)
- Pre-process and cache resized images
- Pre-compute adjacency matrices and save them
- Use faster image loading (PIL instead of cv2, or pre-encoded formats)
- Consider using a faster storage backend (SSD, NVMe)

---

## 3. Memory Bottlenecks

### Location
- Multiple locations across the pipeline

### Issues

1. **Large Image Tensors**
   - Source images: 704×704×3 = 1.49 MB per image
   - Dest images: 540×960×3 = 1.55 MB per image
   - Batch size 12: ~36 MB for images alone

2. **Multiple Masks Per Batch**
   - `N_masks_per_batch=32` masks per sample
   - Each mask: 540×960 = 518K pixels
   - 32 masks × 518K × 1 byte = ~17 MB per sample
   - Batch size 12: ~199 MB for masks

3. **Dense Feature Maps**
   - Stored for both source and destination images
   - Multiple interpolations and transformations
   - Intermediate tensors not freed immediately

4. **Attention Matrices**
   - Cross-attention: `[B, N_desc, N_img]` similarity matrices
   - For 32 masks and 2500 patches: large attention maps

### Impact
- GPU memory pressure
- Potential OOM errors with larger batch sizes
- Slower due to memory allocation/deallocation

### Recommendations
- Use gradient checkpointing for memory efficiency
- Clear intermediate tensors explicitly
- Reduce `N_masks_per_batch` if memory constrained
- Use mixed precision training (FP16/BF16)
- Profile memory usage with `torch.profiler`

---

## 4. Model Architecture Bottlenecks

### Location
- `src/O-MaMa/model/model.py`: `Attention_projector.forward()`
- `src/O-MaMa/model/model_layers.py`: `Context_Attn`

### Issues

1. **Cross-Attention Operations**
   ```python
   # Line 40, 44 in model.py
   Q_context_cross = self.CROSS_context_attn(source_obj, dest_dense_feats, ...)
   T_context_cross = self.CROSS_context_attn(dest_obj, source_dense_feats, ...)
   ```
   - Two cross-attention passes per forward
   - Attention over large feature maps (1,936+ tokens)
   - Complexity: O(N_desc × N_img)

2. **Similarity Computation**
   ```python
   # Line 52 in model.py
   similarity = F.cosine_similarity(Q_desc_norm, T_desc_norm, dim=2)
   ```
   - Computed for all mask pairs
   - For 32 masks: 32×32 similarity matrix per sample

3. **Fixed Position Embeddings**
   - Model has hardcoded position embeddings for specific image sizes
   - Forces image resizing, which may reduce quality

### Impact
- Moderate computational cost
- Memory usage for attention maps

### Recommendations
- Profile attention operations specifically
- Consider using Flash Attention for efficiency
- Optimize similarity computation (maybe use matrix multiplication directly)
- Consider reducing context size or number of masks

---

## 5. I/O and Checkpointing

### Location
- `src/O-MaMa/main.py`: `save_checkpoint()`, `save_json()`
- Evaluation: JSON serialization

### Issues

1. **Frequent Checkpointing**
   - Saves checkpoint after every epoch
   - Model weights: ~42 MB (from Exo2Ego_weights_checkpoint.pt)
   - File I/O can block training

2. **Large JSON Files**
   - Evaluation saves predictions and ground truth as JSON
   - RLE-encoded masks in JSON format
   - Can be large for full datasets

3. **Synchronous I/O**
   - All file operations are blocking
   - No async I/O for checkpoints

### Impact
- Minor, but noticeable during checkpoint saves
- Can cause training stalls

### Recommendations
- Save checkpoints asynchronously
- Use compression for checkpoints
- Consider saving less frequently
- Use binary formats (HDF5, NPZ) instead of JSON for large data

---

## 6. Adjacency Matrix Computation

### Location
- `src/O-MaMa/dataset/adj_descriptors.py`: `get_adj_matrix()`

### Issues

1. **Delaunay Triangulation**
   - Computed for every sample during training
   - Scipy operation, runs on CPU
   - Complexity: O(N log N) where N = number of masks

2. **Second-Order Adjacency**
   - Default `order=2` requires matrix multiplication
   - Additional computation overhead

### Impact
- Moderate CPU bottleneck during data loading
- Adds latency to `__getitem__()`

### Recommendations
- Pre-compute and cache adjacency matrices
- Save as .npy files alongside masks
- Only recompute if masks change
