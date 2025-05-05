
HOW TO RUN
-----------------------------------------------------------------
The project was developed on python 3.10.16 and it is recommended
to use that version if testing the code. It is developed on Ubuntu
through WSL as the LightFM model struggles to compile on Windows,
so if testing it is recommended to be on Linux.

The quicker way to test the code is to create a virtual environment.

Navigate to project directory:

cd fyp

Then create venv:

python -m venv venv

Activate venv:

source venv/bin/activate

The zipped project folder has a requiremnts.txt which has all the
dependencies needed to run the product.

Once you are in the project directory and the virtual environment
is activated you can install all the dependencies by:

pip install -r requirements.txt

-----------------------------------------------------------------
To be able to fetch data from the Riot Games API you will need to 
do two things.

1.Firstly you will need to obtain a free API key from Riot Games:

https://developer.riotgames.com/

As the API key I'm using is not for production I cannot upload it,
sorry for the inconvenience!

2. You need to create an .env file based on the example provided in
.env.example and use your own API key.  

-----------------------------------------------------------------
To start the server on local host and port:5000 run this:

python -m run.main

Then you can go to a browser and reach it with:

http://127.0.0.1:5000/
-----------------------------------------------------------------
