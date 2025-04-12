# Steps to installation
cd (folder path)
python -m venv venv (based on python version installed -  I have used version 3.9)
source venv/bin/activate
pip install -r requirements.txt
streamlit run (main file name).py
Note: As secured files are removed, this app wont run directly until we add the necessary files (.env which has secured keys stored, apple .p8, google json secured files)
