# [CRC Project - IST 2019/2020](https://fenix.tecnico.ulisboa.pt/disciplinas/CRC7/2019-2020/1-semestre/project-1-0f6)
This repository was created to develop both project 1 and project 2 of **Network Science** course.

# Dependencies
## Install python
```bash
# install python
brew install python

# install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user
rm get-pip.py
```

## Install python packages
I recommend creating a new python environment to install all packages required for both projects. This way you will protect both the python's global environment and the environment needed to develop this project. To do this run:
```bash
pip3 install virtualenv
cd <path-to-crc-project-folder>
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

# Run Code
Eact time you want to run python you have to activate the project's environment and then run your script:
```bash
source venv/bin/activate
python run_my_script.py
```
To desactivate the project's private environment just run:
```bash
deactivate
```
