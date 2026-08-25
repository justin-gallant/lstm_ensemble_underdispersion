# Ranked Right, Sized Wrong: Ensemble Spread Fails Silently at Streamflow Extremes

Official code repository and computational pipeline for:
> **Ranked Right, Sized Wrong: Ensemble Spread Fails Silently at Streamflow Extremes**  
> Anonymous Authors
> (Submitted 2026)  

## Citation

**APA:**
> Anonymous Authors. (2026). LSTM Ensemble Under-dispersion Code (Version v1.0.0) [Computer software]. Zenodo. 10.5281/zenodo.22100521

**BibTeX**
```bibtex
@misc{anonymous_lstm_ensemble_underdispersion_2026_zenodo,
  author = {Anonymous Authors},
  title = {LSTM Ensemble Under-dispersion Code},
  month = aug,
  year = 2026,
  copyright = {Creative Commons Attribution 4.0 International},
  howpublished = {Zenodo},
  doi = {10.5281/zenodo.22100521}
  }
```

---

## Overview

This repository implements an evaluation framework to audit the uncertainty signals produced by field-standard deep learning ensembles for streamflow forecasting, specifically assessing their reliability under increasingly frequent hydroclimatic extremes.

Using a 10-member random-initialization Long Short-Term Memory (LSTM) ensemble trained across 269 unregulated catchments in western Canada (1980–2022), we decouple and evaluate two distinct properties of the reported predictive uncertainty:
* **Ensemble Spread ($\hat{\sigma}_{e}^{2}$):** The between-member variance produced by the ensemble, serving as the operational signal of epistemic uncertainty.
* **Dispersion Sizing (Under-dispersion factor):** The ratio of realized residual error to the reported ensemble spread. This framework evaluates how this scaling behaves as flows move from ordinary conditions to unprecedented, beyond-historical-record extremes.
* **Uncertainty Ranking (AUSE):** The Area Under the Sparsification Error, used to determine if the ensemble spread correctly orders predictions by the magnitude of their realized error. We introduce a flow-magnitude baseline ($\hat{\mu}$) to isolate genuine epistemic skill from magnitude-driven metric inflation.

The code in this repository can reproduce all figures, tables, and reported values in the study. All data used is publicly available. Setup instructions are given below.

---

## Local Setup & Preprocessing
Before training on the cluster, it is recommended that the data be downloaded and preprocessed locally. To avoid conflicts with other projects, you should do this within an isolated Python virtual environment.

*Note: If downloading from Zenodo, data are provided and you can skip right to the postprocessing step.*

1. **Create and Activate a Virtual Environment:**
   Ensure you have Python 3.10+ installed. Open your terminal in the project root and run:

   *For Windows (Command Prompt / PowerShell):*
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
   *(Note: If PowerShell returns an "UnauthorizedAccess" or "running scripts is disabled" error, run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as an administrator or for your current user, then try activating again).*

   *For macOS/Linux*
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install Dependencies:**
   With the environment active (you should see `(.venv)` in your terminal prompt), install the training requirements and the local package:

   ```bash
   pip install -r preprocessing_requirements.txt
   pip install -e .
   ```

3. **Run Preprocessing:**
   Launch Jupyter Notebook from your active environment:
   ```bash
   jupyter notebook
   ```
   Open and run `notebooks/01_data_preprocessing.ipynb` in full.
   * Note: This notebook downloads ERA5 reanalysis data which can take a significant amount of time depending on the Copernicus server queues.
   * Outcome: This generates the lightweight CSVs in `data/processed/` required for the training pipeline.

---

**Files of particular note:**
* `data/processed/combined_streamflow.csv`: this is the ground-truth streamflow data in units of $m^3/s$. Note that there will be some gaps in this data.
* `data/processed/static_attributes.csv`: this is the values of static variables for each station. Area is in units of $\mathrm{km}^2$, elevation is in $\mathrm{m}$, and slope is unitless.
* `data/processed/climate/`: this folder contains CSV files of the dynamic variables, structured in the same way as `combined_streamflow.csv`. Temperature has units of degrees celcius while precipitation variables are in units of millimeters averaged over the basin.
* `data/output` contains daily predictions for each model, structured in the same manner as `combined_streamflow` except using units of millimeters over the basin area.
* `src/config.py` is used to consistantly reference common files and directories. Include this in your import statment when developing code or performing analysis. `from src.config import ___`.

---

## Repository Structure

```text
lstm_ensemble_underdispersion/
├── data/
│   ├── output/                  # model predictions of streamflow
│   ├── processed/               # data processed for use in model training and analysis
│   │   ├── climate/                    # daily climate data for each basin
│   │   │   ├── daily_precipitation.csv # daily precipitation by basin
│   │   │   ├── daily_rainfall.csv      # daily rainfall by basin
│   │   │   ├── daily_snowfall.csv      # daily snowfall by basin
│   │   │   ├── daily_temp_max.csv      # daily max temperature by basin
│   │   │   └── daily_temp_min.csv      # daily min temperature by basin
│   │   ├── combined_streamflow.csv     # daily observed mean streamflow per basin
│   │   └── static_attributes.csv       # static attributes for each basin
│   └── raw/                     # raw data downloads (unmodified)
│       ├── dem_data/            # downloaded digital elevation maps
│       ├── drainage_areas/      # drainage area polygons
│       ├── era5/                # temperature and precipitation data downloaded from ERA5
│       │   ├── precipitation/   # raw hourly precipitation data
│       │   └── temperature/     # raw hourly temperature data
│       ├── RGI-western-canada/  # RGI glacier data and polygons
│       ├── spatial_bounds.csv   # latitude and longitude bounds used for downloads
│       └── station_metadata.csv # raw HYDAT station metadata
├── figures/                     # Manuscript figures
├── hpc/                         # HPC submission & environment scripts
│   ├── job.sh                   # Main SLURM execution script
│   ├── setup_env.sh             # Conda environment setup for PyTorch/CUDA
│   └── submit.sh                # Job submission wrapper
├── models/                      # Saved models
├── local_postprocessing/        # Local setup for figures notebook
│   ├── clean_local.sh           # Cleanup script for local cached evaluation runs
│   └── setup_local.sh           # Setup script for local analysis environment
├── notebooks/                   # Jupyter notebooks for data processing & paper figures
│   ├── 01_data_preprocessing.ipynb # Gauge filtering, ERA5-Land extraction, & attribute building
│   └── 02_workshop_figures.ipynb   # Main manuscript figure generation
├── src/                         # Core Python package
│   ├── __init__.py
│   ├── climate.py               # ERA5-Land climate processing routines
│   ├── config.py                # Central experiment configuration & default paths
│   ├── data_ingestion.py        # HYDAT hydrometric data readers
│   ├── data_utils.py            # Normalization, sequence building, & masking utilities
│   ├── dataset.py               # PyTorch Dataset classes for sequence windows
│   ├── inference.py             # Factual and Counterfactual (g=0) prediction engine
│   ├── models.py                # Regional LSTM architecture definitions
│   ├── processing.py            # Data pipeline orchestration
│   ├── spatial_utils.py         # Geospatial intersections & polygon operations
│   └── training.py              # Loss functions (Masked NSE*) and training loop
├── .gitignore
├── bundle_project.py            # Utility to bundle project dependencies for HPC transfer
├── preprocessing_requirements.txt  # Dependencies for local file downloading and preprocessing
├── postprocessing_requirements.txt # Dependencies for local analysis & figure generation
├── requirements.txt             # Primary PyTorch training dependencies for HPC
├── README.md
├── run_training.py              # Main CLI entry point for training and evaluation
├── secrets.env                  # Email and allocation code for HPC setup
└── setup.py                     # Local package installation setup
```

---

## Model & Experiment Configurations
The `run_training.py` pipeline supports multiple architectures, loss functions, and input feature sets to facilitate broad experiment testing. The specific setup used in the paper was the combination of `lstm`, NSE*, and `topographic` input set.

1. **Model Architectures** (`<model_type>`):
* `lstm`: A standard Long Short-Term Memory network where static catchment attributes are concatenated with dynamic meteorological inputs at each timestep.
* `ealstm` An Entity-Aware LSTM where static catchment attributes are explicitly separated from dynamic inputs to modulate the input gate.

2. **Loss Functions**:
* NSE* (`BasinAveragedNSELoss`): A basin-averaged Nash-Sutcliffe Efficiency loss function. This normalizes the loss across catchments to prevent high-variance basins from dominating the optimization. (Set as default in `run_training.py`)
* MSE (`MaskedMSELoss`): A standard Mean Square Error loss function. (To use this, modify the `criterion` variable inside `run_training.py`)

3. **Input Feature Configurations** (`<exp_name>`)
* `baseline`:
  * Dynamic: `temp_max`, `temp_min`, `precip`
  * Static: `glacier_pct`
* `area`:
  * Dynamic: `temp_max`, `temp_min`, `precip`
  * Static: `basin_area_km2`, `glacier_pct`
* `topographic`:
  * Dynamic: `temp_max`, `temp_min`, `precip`
  * Static: `basin_area_km2`, `mean_elev`, `elev_range`, `mean_slope`, `glacier_pct`
* `phase-split`:
  * Dynamic: `temp_max`, `temp_min`, `rain`, `snow`
  * Static: `basin_area_km2`, `mean_elev`, `elev_range`, `mean_slope`, `glacier_pct`

---

## High Performance Compute Setup
To train the LSTM model, this project utilizes a SLURM-based HPC cluster. The following steps can be followed to set up this project:

1. **Create a Secrets File**
   Create a file named `secrets.env` in the project root containing your email and allocation code.
   ```text
   # secrets.env
   EMAIL="<your email>"
   ACCOUNT="<alloc-code>-gpu"
   ```
   Note: The -gpu suffix is generally required for running in a GPU environment, depending on your cluster's configuration.
2. **Bundle the Project**
   Run the bundle script, `python bundle_project.py`, in your terminal from the project root. This creates a zip file excluding the raw data.
3. **Upload to Cluster**
   Run `scp project_upload.zip <username>@<cluster-address>:/scratch/<alloc-code>/` replacing `<username>` with your login and `<alloc-code>` with your allocation code.
4. **Connect and Extract**
   SSH into the cluster and unzip the project.
   ```bash
   ssh <username>@<cluster-address>
   cd /scratch/<alloc-code>
   unzip project_upload.zip -d lstm_ensemble_underdispersion
   cd lstm_ensemble_underdispersion
   ```
5. **Setup Environment (One-time)**
   This script loads the required Python modules, creates a virtual environment, and installs dependencies.
   ```bash
   chmod +x hpc/setup_env.sh
   ./hpc/setup_env.sh
   ```
   *Troubleshooting*: If you recieve an error, try running this first and then trying again:
   ```bash
   sed -i 's/\r$//' setup_env.sh
   sed -i 's/\r$//' submit.sh
   sed -i 's/\r$//' job.sh
   ```
6. **Submit the Job**
   The submit script automatically handles directory setup, secrets injection, and SLURM submission. You must provide the experiment name and model type as command-line arguments. 
   ```bash
   chmod +x hpc/submit.sh
   ./hpc/submit.sh <experiment_name> <model_type>
   ```
   (Example: `./hpc/submit.sh topographic lstm`) 

### Monitoring and Results
* **Check Status:** Run `squeue -u <username>` to see your job in the queue.
* **View Logs:** Once running, track progress live: `tail -f logs/train_*.out`
* **Retrieve Results:**
   Once the job status has changed to `COMPLETED`, you can download the trained model and the factual/counterfactual predictions to your local machine. Saved files can be found at the following locations, with `member_id` corresponding to each of the 10 ensemble runs (0-9):
   * Trained models: `models/<exp_name>_<model_type>_member_<member_id>.pth` (*Example:* `topographic_lstm_member_0.pth`)
   * Predictions: `data/output/<exp_name>_<model_type>_preds_member_<member_id>.csv` (*Example:* `topographic_lstm_preds_member_0.csv`)
   The following code will download the necessary files:
   ```bash
   # download the predictions
   scp -r <username>@<cluster-address>:/scratch/<alloc-code>/lstm_ensemble_underdispersion/data/output/* ./data/output/
   ```
   ```bash
   # download saved models
   scp -r <username>@<cluster-address>:/scratch/<alloc-code>/lstm_ensemble_underdispersion/models/* ./models/
   ```

---

## Postprocessing & Figure Generation
After retrieving the factual and counterfactual predictions from the HPC cluster (or downloading the paper results from Zenodo), use the local postprocessing environment to generate the manuscript figures.

### Option 1: `venv` (Recommended for Windows)
1. Create the virtual environment:
```bash
python -m venv ensemble-postprocessing-env
```
2. Activate the environment:
  * PowerShell: `.\ensemble-postprocessing-env\Scripts\Activate.ps1`
  * Command Prompt: `.\ensemble-postprocessing-env\Scripts\activate.bat`
3. Upgrade `pip` and install the required dependencies:
```bash
python -m pip install --upgrade pip
pip install -r postprocessing_requirements.txt
```
4. Install `ipykernel` and register the environment:
```bash
pip install ipykernel
python -m ipykernel install --user --name ensemble-postprocessing-env --display-name "Ensemble Postprocessing"
```
5. Install the local project configuration:
```bash
pip install -e .
```

### Option 2: Automated via `pyenv` (Recommended for macOS & Linux)
*Prerequisite: Ensure `pyenv` and `pyenv-virtualenv` are installed on your system.*

An automated bash script is provided to handle the entire lifecycle of the environment, including Python installation, environment creation, dependency installation, and kernel registration.

1. Open your terminal and navigate to the local postprocessing directory:
```bash
cd local_postprocessing
```
2. Make the setup script executable (It should already be, but just in case):
```bash
chmod +x setup_local.sh
```
3. Run the setup script:
```
./setup_local.sh
```
*(Note: To completely remove the kernel and virtual environment from your system later, you can run `./clean_local.sh`)*

### Run the figure notebook
Once the setup is complete, open `notebooks/02_workshop_figures.ipynb` and select your new environment from the kernel dropdown menu. This notebook processes the downloaded prediction arrays and reproduces all figures and performance summary tables from the manuscript.

---

## Funding & Acknowledgements
[Removed for double-blind review.]