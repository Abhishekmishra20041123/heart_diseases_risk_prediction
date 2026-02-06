@echo off
echo Starting Heart Disease Detection Dashboard...
echo.
echo Installing required packages...
pip install -r requirements.txt
echo.
echo Opening dashboard in your browser...
echo.
python -m streamlit run dashboard.py
pause