# Personal Health Information Blur
Program blurs out parts of an image that may reveal personal health information using a combination of detection models.

## Setup
Requires python versions **3.8.10** and **3.7.4**

Each folder in the models directory requires a venv of it's own, since each model requires specific libraries to run, use the exact python.exe from the source folder for your python installations. On Windows you can find this by running the following in command prompt.
` where python `

for each model, cd into the corresponding folder and run 
```
[path_to_specific_version_python_exe] -m venv [venv_name] 
venv_name\Scripts\activate 
pip install --upgrade pip 
pip install -r requirements.txt
```

Follow this list when deciding which version of python to use for each model
3.8.10
easyocr
insightface
mtcnn

3.7.4
craft
paddle_ocr
retina_face

## Running the main.py

` python main.py [input folder] [output_folder] `
