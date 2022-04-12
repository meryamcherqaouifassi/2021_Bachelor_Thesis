#!/bin/bash

#slurm batch commands set up the script to be run as a slurm batch job.
#The job is therefore not interactive - it's shoot and forget

#SBATCH --mail-type ALL 
#SBATCH --mail-user milton.gomez@unil.ch

#changing the working directory to where I will save your data
#SBATCH --chdir ./data
#SBATCH --job-name meryam_dl
#SBATCH --output output_%A_%a.out
#SBATCH --error error_%A_%a.err

#SBATCH --partition cpu

#SBATCH --ntasks 1 
#SBATCH --cpus-per-task 1 
#SBATCH --mem 1G 
#SBATCH --time 08:00:00 
#SBATCH --export NONE

#clearing modules and loading default module(s)
module purge
module load gcc

#set up the time variables we will use to contruct the URL we will download (end_day, year, month) and that will
#give our output files their names. If you want to improve the script, you can set up arrays of stations/regions
#to download all the stations you're interested in in one go.


#set up an array with month end date. That way we can download the proper length of data since the website
#doesn't handle "beyond maximum" time well. (if you ask for data "up to Feb. 31st" it just returns an error
#instead of returning data up to Feb. 28th since max is beyond bounds.  
declare -A end_day
end_day=( [01]=31 [02]=28 [03]=31 [04]=30 [05]=31 [06]=30 [07]=31 [08]=31 [09]=30 [10]=31 [11]=30 [12]=31) 


for year in {1973..2021}
do
	for month in {01..12}
	do
            #I got the URL from the website and saw that for Casablanca, all I had to do was vary the YEAR and MONTH in the URL.
	    #That's why we're iterating through every year and month. Note that some months may have no data! (data holes)

	    #Build the URL
	    URL='weather.uwyo.edu/cgi-bin/sounding?region=africa&TYPE=TEXT%3ALIST&YEAR='$year'&MONTH='$month'&FROM=0100&TO='${end_day[$month]}'12&STNM=60155'

	    #Build the filename for our output file. Casablanca_YYYY_MM.txt. Saved as a TXT even though originally it's HTML.
	    filename='Casablanca_'$year'_'$month'.txt'
	    wget $URL -O $filename

	    #one second break for the server. Because we want to be nice and not be considered an attack :) (setting lower sleep
	    #times results in the server sometimes serving empty documents)
	    sleep 1s	
	done
done

