@echo off
REM =================================================================
REM  Batch file to start the Image Captioning Web App
REM =================================================================

ECHO Starting the Image Captioning application...

REM Activate the Conda environment
ECHO Activating Conda environment 'tf_env'...
CALL conda activate tf_env

REM Check if activation was successful
IF %CONDA_DEFAULT_ENV% NEQ tf_env (
    ECHO ERROR: Failed to activate the 'tf_env' Conda environment.
    ECHO Please make sure Conda is installed and the environment exists.
    PAUSE
    EXIT /B
)

REM Install required packages (optional, can be commented out after first run)
ECHO Installing dependencies from requirements.txt...
pip install -r requirements.txt

REM Run the Flask application
ECHO Starting the Flask server...
python frontend/src/app.py

REM Keep the window open after the server stops
PAUSE
