REM Script to run activate_app.py and open browser with right site
REM Then wait on enter to start eye tracker process

cd Activate\
start cmd /k C:\Users\Helen\PycharmProjects\venv\Python38\Scripts\python.exe C:/Users/Helen/Documents/DSP-Project/Activate/activate_app.py
REM Pause for python init
timeout 10 /nobreak
start "" http://127.0.0.1:7234/

REM Starting eye tracker
REM start /min ..\EyeTrackerCode\main.exe ^&

exit
