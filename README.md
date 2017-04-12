Currently only contains an offline version.


`offline/run_tasks.py`, from where you can run any functions necessary. Code is stored in `


### General notes

## Introduction

# Setup
0. Purchase Mac.

1. Create a virtualenv. Make sure your python3 is python3.6 (or latest version)

    `$ virtualenv --python=/usr/local/bin/python3 env`

2. Activate virtualenv.

	`$ source env/bin/activate` 

3. Install all packages using pip.
    
    `$ pip install -r requirements.txt`

4. Create database tables.

    `$ python offline/application/models/quanta.py`

# Structure

* The directory `offline/application/` contains all of the required source code.
* The raw dataset is stored in `offline/application/csv`.
* The database structure is stored in `offline/application/models/`.
* Tasks are stored in `offline/application/tasks`.
* General tools are stored in `offline/application/utils`.
* Outputs are stored in `results/`.


## Usage

Use `run_tasks.py` to manage the application. First, populate the database using the `load_all_data()` task. Then you can run the optimizer tasks as you see fit. Run `get_portfolio_and_plot_results()` to plot outputs.




