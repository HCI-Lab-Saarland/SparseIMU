# SparseIMU: Computational Design of Sparse IMU Layouts for Sensing Fine-Grained Finger Microgestures (ACM TOCHI 2023, Released Code)

**[Adwait Sharma](https://www.adwaitsharma.com/)**, **[Christina Salchow-Hömmen](https://orcid.org/0000-0001-5527-9895)**, **[Vimal Suresh Mollyn](https://vimal-mollyn.com/)**, **[Aditya Shekhar Nittala](https://orcid.org/0000-0002-3698-9733)**, **[Michael A. Hedderich](https://www.michael-hedderich.de/)**, **[Marion Koelle](https://marionkoelle.de/)**, **[Thomas Seel](https://orcid.org/0000-0002-6920-1690)**, **[Jürgen Steimle](https://hci.cs.uni-saarland.de/people/juergen-steimle/)**

![SparseIMU Teaser Image](https://github.com/HCI-Lab-Saarland/SparseIMU/blob/main/media/SparseIMU_teaser.png)

### Abstract
Gestural interaction with freehands and while grasping an everyday object enables always-available input. To sense such gestures, minimal instrumentation of the user’s hand is desirable. However, the choice of an effective but minimal IMU layout remains challenging, due to the complexity of the multi-factorial space that comprises diverse finger gestures, objects, and grasps. We present SparseIMU, a rapid method for selecting minimal inertial sensor-based layouts for effective gesture recognition. Furthermore, we contribute a computational tool to guide designers with optimal sensor placement. Our approach builds on an extensive microgestures dataset that we collected with a dense network of 17 inertial measurement units (IMUs). We performed a series of analyses, including an evaluation of the entire combinatorial space for freehand and grasping microgestures (393 K layouts), and quantified the performance across different layout choices, revealing new gesture detection opportunities with IMUs. Finally, we demonstrate the versatility of our method with four scenarios.

[Read full article](https://dl.acm.org/doi/full/10.1145/3569894)

---

## Computational Design Tool for Rapid Selection of Custom Sparse Layouts

![SparseIMU Tool](https://github.com/HCI-Lab-Saarland/SparseIMU/blob/main/media/SparseIMU_tool.gif)

---

## Installation Instructions

### Step 1: Clone the repository
```bash
git clone https://github.com/HCI-Lab-Saarland/SparseIMU.git
```

### Step 2: Download the processed dataset

Please download the 'processed_dataset' from [this](https://hci.cs.uni-saarland.de/wp-content/uploads/projects/micro-gestural-input/sparseimu/processed_dataset.zip) (approx. 250 MB) link, which is used in the tool, and save it to the SparseIMU directory. The raw dataset link is provided below, along with details of the processing pipeline and tool architecture in the article.

### Step 3: Set up the environment and install dependencies
We recommend using Conda. The setup has been tested on MacBook (MacOS 12) with `Python 3.7`.

```bash
conda create -n "sparseimu" python=3.7
conda activate sparseimu
conda install pandas==1.2.1 flask==1.1.2 matplotlib=3.3.2 seaborn=0.11.1 pytables=3.6.1 scikit-learn==0.24.1 
```

### Step 4: Run the tool
```bash
python sparseimu/tool.py
```
> If the browser window doesn't open automatically, use the IP and port displayed in the terminal (e.g., `http://localhost:4444`).

---

## Raw dataset

Download the raw dataset [here](https://hci.cs.uni-saarland.de/wp-content/uploads/projects/micro-gestural-input/sparseimu/SparseIMU_fulldataset.zip) (approx. 9GB).

---

## License

This project is licensed under the MIT License.

---

## Citation
If you find our article or released code/dataset useful, please cite our work:

```bibtex
@article{10.1145/3569894,
author = {Sharma, Adwait and Salchow-H\"{o}mmen, Christina and Mollyn, Vimal Suresh and Nittala, Aditya Shekhar and Hedderich, Michael A. and Koelle, Marion and Seel, Thomas and Steimle, J\"{u}rgen},
title = {SparseIMU: Computational Design of Sparse IMU Layouts for Sensing Fine-grained Finger Microgestures},
year = {2023},
issue_date = {June 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {30},
number = {3},
issn = {1073-0516},
url = {https://doi.org/10.1145/3569894},
doi = {10.1145/3569894},
journal = {ACM Trans. Comput.-Hum. Interact.},
month = {jun},
articleno = {39},
numpages = {40},
keywords = {Gesture recognition, hand gestures, sensor placement, imu, objects, design tool}
}
```

---

## Contact

Please contact [Adwait Sharma](https://www.adwaitsharma.com/) if you have any questions about SparseIMU for your use.

