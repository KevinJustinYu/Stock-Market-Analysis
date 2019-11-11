@ECHO OFF 
TITLE Executing daily_csv_update.py in an anaconda environment (tensorflow_env)
ECHO Please Wait...
:: Section 1: Activate the environment.
ECHO ============================
ECHO Conda Activate
ECHO ============================
@CALL "C:\Users\kevin\Anaconda3\Scripts\activate.bat" tensorflow_env
:: Section 2: Execute python script.
ECHO ============================
ECHO Python daily_csv_update.py
ECHO ============================
python "C:\Users\kevin\Documents\Projects\Coding Projects\Stock Market\Stock-Market-Analysis\daily_csv_update.py"

ECHO ============================
ECHO End
ECHO ============================

PAUSE