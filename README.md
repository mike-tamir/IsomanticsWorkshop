# Isomantics workshop
This repo was prepared for the Isomantics workshop Nov 2017.
* The main workshop notebook is located here
  * `code/Isomantics_workshop_driver_notebook.ipynb`

## Setup
In order to run the notebook you should
1. Clone or download this repo.
2. Open terminal and navigate to the repo root folder with
  * `$ cd path/to/repo/root`
3. Run the following in the command line
  * `$ sudo bash isomantics_setup.sh`
    * This script uses anaconda to set up a virtual environment on your local machine.
    * Enter your password when prompted to run the setup.
    * To activate the local environment run in terminal:
      * `source activate isomantics`
      * you should see `(isomantics)` prepending on your terminal if activated:
        * `(isomantics) username$`
    * To deactivate the environment run in terminal:
      * `source deactivate`
  * Congratulations! Now your virtual environment is set up with everything you will need for the isomantics workshop.
4. Launch jupyter notebook by typing the following code in the command line:
  * `jupyter notebook`
5. Navigate to the `code/` dir and open `code/Isomantics_workshop_driver_notebook.ipynb` by clicking on the link.
