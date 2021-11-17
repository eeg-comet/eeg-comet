# Welcome to Microstate Tools

Authors: Amin Kabir, Raaj Chatterjee

This git repository is for developing and publishing new versions of the app.

To download the stand-alone versions of the app for your OS, please visit our google drive folder [linked here](https://drive.google.com/drive/folders/1W786fr4vwZJAcKBJHkQbKZQ7R8S9G0y7?usp=sharing).

# Setup and Run the program

Using a virtual environment:
The easiest (and non-intrusive) way of installing fbs is via pip and a Python virtual environment. To create a virtual environment in the current directory, execute the following command:

python3 -m venv venv
Then, activate the environment with one of the commands below:

# On Mac/Linux:
source venv/bin/activate
# On Windows:
call venv\scripts\activate.bat

To install necessary packages:
After cloning the repository, navigate to the repository in terminal

First ensure that you have the most recent version of pip3 and that wheel is installed:
    
    pip3 install --upgrade pip3
    pip3 install wheel

Then install the remaining requirements:

    pip3 install -r requirements/base.txt

Please do not change or upgrade the python package dependencies.

This app uses the fman build system, for more information, please visit: https://github.com/mherrmann/fbs-tutorial

## To run the program, type: 

    fbs run
This should run instantly and not return any errors

# Packaging the App

## To build the program, type:
    fbs freeze

To see the verbose version:

    fbs freeze --debug

This creates the folder `target/YourApp`. You can copy this directory to any other computer (with the same OS as yours) and run the app there!

## To package the program into an installer, follow the instructions:
fbs lets you generate each of the above packages via the command:

    fbs installer

Depending on your operating system, this may require you to first install some tools. Please read on for OS-specific instructions.

### For windows:

Before you can use the `installer` command on Windows, please install [NSIS](http://nsis.sourceforge.net/Main_Page) and add its installation directory to your `PATH` environment variable.

The installer is created at `target/YourAppSetup.exe`. It lets your users pick the installation directory and adds your app to the Start Menu. It also creates an entry in Windows' list of installed programs. Your users can use this to uninstall your app.


### For mac:

On Mac, the `installer` command generates the file `target/YourApp.dmg`
To install your app, your users simply open the .dmg file, then drag the app's icon to the Applications folder

### For linux:

On Linux, the installer command requires that you have [fpm](https://github.com/jordansissel/fpm). You can for instance follow [these instructions](https://fpm.readthedocs.io/en/latest/installation.html) to install it.

Depending on your Linux distribution, fbs creates the installer at target/YourApp.deb, ...pkg.tar.xz or ...rpm. Your users can use these files to install your app with their respective package manager.
