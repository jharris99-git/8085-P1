# COMP 8085 Project 1

## Installation

1. Open a terminal in the project root directory.
2. Run `py -m venv ./p1`
3. Set `./p1/Scripts/python.exe` as the python interpreter in your dev environment.
4. Run `p1/Scripts/activate`
5. Run `python -m pip install -r requirements.txt`
6. Run `deactivate`.

Linux instructions:
1. 

## Using the Project

#### Start

1. Open terminal in the project root directory.
2. Run `p1/Scripts/activate` or `p1\Scripts\activate` to activate the virtual environment.
3. Run `python -m pip install -r requirements.txt` to make sure your local venv is up-to-date.

#### Execution

1. Make sure the above commands have been executed.
2. Run `python -m src/NIDS.py`
\* Note that different operating systems may use different slash separators/

#### Close

1. Open terminal in the project root directory.
2. Run `pip freeze > requirements.txt` to add any possible new requirements.
3. Run `deactivate` to exit the virtual environment.

## Environment Structure

- `/datasets` contains the original dataset as well as prepared datasets for use in training.
- `/p1` contains the virtual environment files. This is local and should not be committed to the repository.
- `/src` contains the python source files for the project.
- `/models` contains serialized binaries of the preset model weights.
- `/test_data` is where input data csv files should be stored.
