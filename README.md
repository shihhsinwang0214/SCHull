# SCHull

**SCHull** (Sparse, Connected, and Rigid Graph Construction) is a geometry-aware method for building molecular and point-cloud graphs that are sparse, connected, and rigid — ideal for use in 3D geometric learning tasks such as molecular modeling and protein structure representation.

This repository supports the ICLR 2025 Oral Presentation:  
**[A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules](https://openreview.net/forum?id=OIvg3MqWX2)**  
**Wang, S. H.\*, Huang, Y.\*, Baker, J., Sun, Y. E., Tang, Q., Wang, B.**

📄 [Slides](https://shihhsinwang0214.github.io/persnoal_website/assets/ICLR25/ICLR_2025_oral_slides.pdf) | 🧵 [Poster](https://shihhsinwang0214.github.io/persnoal_website/assets/ICLR25/ICLR_2025_oral_poster.pdf) 

💻 Code will be available soon as a pip-installable library!

---

## 🚀 Getting Started

### 🔧 Clone the repository

```bash
git clone https://github.com/shihhsinwang0214/SCHull.git
```

### 🧱 Import SCHull

```python
from SCHull import SCHull
```

---

## 📌 Usage: `get_schull`

### Inputs

```python
SCHull.get_schull(positions: np.ndarray, atom_types: np.ndarray)
```

- `positions`: NumPy array of shape `(N, 3)` for 3D coordinates.
- `atom_types`: NumPy array of shape `(N,)` with atom type labels (or placeholders).

### Outputs

Returns:
- `positions`: the original input.
- `atom_types`: the original input.
- `edge_index`: source-target index pairs.
- `edge_attr`: edge features (length, angle).
- `radial_arr`: radial distances.

---

## 🧪 Example

```python
import numpy as np
from torch_geometric.data import Data
from SCHull import SCHull

pos = np.random.rand(10, 3) * 10
atoms = np.zeros(10, dtype=int)

schull = SCHull()
_, _ , edge_index, edge_attr, radial_arr = schull.get_schull(pos, atoms)

data = Data(atoms=atoms, edge_index=to_undirected(edge_index), pos=pos, natoms= 10, cell=cell, edge_attr= edge_attr)
```


---

## 📌 About

This project is developed as part of ongoing research in symmetry-aware geometric deep learning. It was proposed and led by Shih-Hsin Wang and received an **Oral Presentation** at ICLR 2025 (top 1.8%).

---

## ✨ Coming Soon

- `pip install schull` support
- Integration with popular molecular datasets
- Prebuilt graph visualizations

---

## 🧠 Citation

If you find this code useful, please cite:

```
@inproceedings{wang2025schull,
  title={A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules},
  author={Wang, Shih-Hsin and Huang, Yuhao and Baker, Justin and Sun, Yuan-En and Tang, Qi and Wang, Bao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

---

📫 Questions or suggestions? Feel free to open an issue or email me at `shwang@math.utah.edu`.

```

Let me know if you want this auto-pushed into your GitHub `README.md` or would like a second section for installation or testing notebooks!
