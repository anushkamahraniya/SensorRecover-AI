# SensorRecover AI

**Missing Sensor Pattern Reconstruction Using Hopfield Networks**

A Hopfield associative memory that restores missing or corrupted readings in real air-quality sensor data, with an interactive Streamlit website.

## Run it

Python 3.10 or newer is needed. From this folder, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The website opens at http://localhost:8501.

The dataset and the benchmark results are already included, so nothing else is needed. To check or repeat the work:

```powershell
python -m pytest tests -q        # automated tests
python run_experiments.py        # repeats tuning and the benchmark; takes about 30 minutes
```

## Put it online for free

1. Create an empty public repository at https://github.com/new, then push this folder to it.
2. Sign in at https://share.streamlit.io with GitHub and select **Create app**.
3. Choose the repository, branch `main`, main file `app.py`. Under advanced settings, choose Python 3.11.
4. Select **Deploy**. The first build takes a few minutes.

## Dataset

UCI Air Quality: 9,357 hourly records from a field deployment in Italy, March 2004 to April 2005.

- Page: https://archive.ics.uci.edu/dataset/360/air+quality
- Download: https://archive.ics.uci.edu/static/public/360/air+quality.zip
- DOI: 10.24432/C59K5F, licence CC BY 4.0

No data is invented. Natural gaps stay empty and are never scored. Scores use only readings that exist in the data and were hidden on purpose.

## Website pages

| Page | What it does |
|---|---|
| Home | Project, problem, how Hopfield networks work, architecture |
| Dataset explorer | Load UCI or upload a CSV; preview, gaps, correlation, distributions |
| Missing data simulator | Choose sensors, missing rate and corruption, then see the damage |
| Hopfield reconstruction | Run the network; compare original and reconstructed values; watch energy fall |
| Model performance | Benchmark tables and charts for all methods, and a live benchmark |
| Visualization dashboard | All charts for the latest reconstruction |
| Download results | Reconstructed CSV, scores, charts, and the report |

## Project layout

```
SensorRecover-AI/
    app.py                     website entry point and navigation
    run_experiments.py         reproduces every reported number
    requirements.txt
    .streamlit/config.toml     pastel blue theme
    assets/logo.svg
    data/AirQualityUCI.csv     original UCI file
    docs/PROJECT_REPORT.md     full academic report and evaluation Q&A
    models/hopfield.py         continuous and binary Hopfield networks
    app_pages/                 one file per website page
    results/                   saved benchmark output
    tests/                     automated tests
    utils/
        preprocessing.py       loading, cleaning, splitting, damage
        evaluation.py          baselines, metrics, tuning, benchmark
        visualization.py       charts
        state.py               data shared between pages
        ui.py                  theme and layout helpers
```

## Troubleshooting

**`DLL load failed ... An Application Control policy has blocked this file`.** Windows Smart App Control can block a newly installed package the first time it loads. On the machine this was built on, scikit-learn loaded normally on the next attempt, so try the command again. If it keeps happening, check Windows Security, and do not switch protection off without considering the risk.

**`Activate.ps1 cannot be loaded`.** Run `Set-ExecutionPolicy -Scope Process RemoteSigned` in that PowerShell window, then activate again. This lasts only for that window.

**`streamlit` is not recognised.** Use `python -m streamlit run app.py`.

**Port 8501 is in use.** Run `python -m streamlit run app.py --server.port 8502`.

**The first reconstruction is slow.** Models are prepared on first use, then reused. An uploaded file also goes through a short tuning step.

**Uploaded CSV rejected.** It needs at least 100 rows, two numeric columns, and some rows where every sensor is present. If your file marks failed readings with a code such as -999, enter it in the upload form.

Report: [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)
