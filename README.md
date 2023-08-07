# Traffic-Accident-Modeling

## Download Resources

1. From OneDrive link, download the `resources` folder
2. `Add` contents of subfolder `distance` and `severity` from the downloaded `resources` folder to this project's `WebApp/resources/{distance/severity}` folder

## Install Dependencies
Execute the appropriate commands below depending on your machine

### Windows
    $ cd WebApp
    $ python -m pip install -r requirements-windows.txt
    
### MacOS
    $ cd WebApp
    $ conda install -c conda-forge shap
    $ python -m pip install -r requirements-macOS.txt
  
## To Run

    $ cd WebApp
    $ python index.py
  
## Tested Environments

|OS   |Python   |Working|
|---|---|---|
|Windows 10   |3.9.13   |:white_check_mark:|
|macOS Ventura   |3.10.6   |:white_check_mark:|
