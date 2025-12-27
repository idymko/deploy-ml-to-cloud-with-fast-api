# Github repository link

https://github.com/idymko/deploy-ml-to-cloud-with-fast-api.git

* Trained Random Forest Classifier on on publicly available Census Bureau data: https://archive.ics.uci.edu/dataset/20/census+income
* Performed testing
* Implemented CI with GitHub Actions
* Implented GET and POST Fast APIs
* Implemented CD with Render

# Intro 

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

In this project, you will apply the skills acquired in this course to develop a classification model on publicly available Census Bureau data. You will create unit tests to monitor the model performance on various data slices. Then, you will deploy your model using the FastAPI package and create API tests. The slice validation and the API tests will be incorporated into a CI/CD framework using GitHub Actions.

Two datasets will be provided in the starter code on the following page to experience updating the dataset and model in git.

# Environment Set up
* **Option 1: Using pip and venv**
    * Ensure you have Python 3.13 installed
    * Create virtual environment: `python3.13 -m venv .venv`
    * Activate environment: `source .venv/bin/activate` (On Windows: `.venv\Scripts\activate`)
    * Install dependencies: `pip install -r starter/requirements.txt`

* **Option 2: Using conda  (Recommended on Mac)**
    * Download and install conda if you don't have it already.
    * `conda create -n deploy-ml "python=3.13" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn pydantic httpx matplotlib seaborn -c conda-forge`
    * Install git either through conda ("conda install git") or through your CLI, e.g. sudo apt-get git.
    * `conda activate deploy-ml`
    * `conda install pip`
    * `pip install -r requirements.txt`

## Setup DVC 
* `dvc init`
* Create local remote folder `mkdir ../local_remote`
* `dvc remote add -d localremote ../local_remote`
* `dvc exp run`
* Change parameters on the fly: dvc exp run --set-param train.n_estimators=50
* Show experiment results: `dvc exp show`
* Push data to remote: `dvc push`

## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs `pytest` and `flake8` on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to use Python 3.13 (same version as development).
    * Note: Add flake8 to requirements.txt if you want to use it for linting: `pip install flake8`

# Data
* Download census.cs (https://archive.ics.uci.edu/dataset/20/census+income) and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.



# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment
* Deploy a Web Service from Render Console
    * Create a free Render account (https://render.com/).
    * After you log in to your account, click New Web Service.
    * You will be prompted to connect your project's GitHub or Gitlab repo and branch to the Web Service.
    * Configure Build and Start Command
        * Build Command: `pip install -r requirements`
        * Start Command: `uvicorn api:app --host 0.0.0.0 --port 10000`


* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
    * Note: Install flake8 separately if needed: `pip install flake8`
* Write a script that uses the requests module to do one POST on your live API.
