#Welcome to Microstate Tools

Authors: Amin Kabir, Raaj Chatterjee

This git repository is for developing and publishing new versions of the app.

# Setup and Run the program

To Initialize the development environment:
After cloning the repository, navigate to the repository in terminal
Please ensure you are running a virtual environment with python version 3.6

Type in:

    pip install -r requirements

Please do not change or upgrade the python package dependencies.

This app uses the fman build system, for more information, please visit: https://github.com/mherrmann/fbs-tutorial

##To run the program, type: 

    fbs run
This should run instantly and not return any errors

# Packaging the App

##To build the program, type:
    fbs freeze

To see the verbose version:

    fbs freeze --debug

This creates the folder `target/YourApp`. You can copy this directory to any other computer (with the same OS as yours) and run the app there!

##To package the program into an installer, follow the instructions:
fbs lets you generate each of the above packages via the command:

    fbs installer

Depending on your operating system, this may require you to first install some tools. Please read on for OS-specific instructions.

###For windows:

Before you can use the `installer` command on Windows, please install [NSIS](http://nsis.sourceforge.net/Main_Page) and add its installation directory to your `PATH` environment variable.

The installer is created at `target/YourAppSetup.exe`. It lets your users pick the installation directory and adds your app to the Start Menu. It also creates an entry in Windows' list of installed programs. Your users can use this to uninstall your app.


###For mac:

On Mac, the `installer` command generates the file `target/YourApp.dmg`
To install your app, your users simply open the .dmg file, then drag the app's icon to the Applications folder

###For linux:

On Linux, the installer command requires that you have fpm. You can for instance follow these instructions to install it.

Depending on your Linux distribution, fbs creates the installer at target/YourApp.deb, ...pkg.tar.xz or ...rpm. Your users can use these files to install your app with their respective package manager.
