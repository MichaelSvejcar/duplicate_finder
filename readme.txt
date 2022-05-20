------------------------------DuplicateFinder-----------------------------------
This project is a system designed for recognising duplicate files in a music database. It can find not only exactly the same audio files, but is robust even to differencies like different bitrates, slight pitch or tempo variations and can even recognise duplicates containing noise or applause at the beggining/end of one of the audio files.
For the evaluation, it is using image hashing with combination of MrMsDTW (both or only of these techniques, depending on accuracy setting). It is built in a way so that it is simple to use for the end user, with four pre-defined accuracy settings, but various parameters can be set if there is need for some more fine-tuning.


--------- HOW TO USE (simple way with pre-defined accuracy settings) -----------
The easiest way to run the system is to set the input parameters in "config.ini" file and run the "run.py" script. The individual parameters are described below:

path - absolute path to directory with music files

outputpath - absolute path to output directory, which is used for saving the final output files in .xlsx and .csv format and files needed for calculation. If outputpath is not set, folder gets created in the path directory automatically.

accuracy - Calculation accuracy, can be set to "low", "normal", "high" and "extreme"; "low" for finding duplicates which are exactly identical, "normal" for cases where there may be some sort of noise in the beggining/end of one duplicate or for case when one duplicate is encoded into very low bitrate, "high" is similar as normal, but has even lower tolerance for differences, "extreme" can be used for cases when user expects very long passages (like half of the whole recording) of noise in beggining/end of some of the audio duplicates.

chroma_method - Method for chroma features calculation, "cuda" is for nnAudio implementation with the use of CUDA, "synctoolbox" uses functions from same-named library without CUDA - dependency


----- HOW TO USE (in a way so that individual parameters can be set manually) ----
If you want to fine-tune the system input parameters, you can use one of the functions below. All of these functions are in the duplicate_finder.py file, with their description and parameters documentation:

find_duplicates(path, outputpath, accuracy, chroma_method)
find_duplicates_dtw(path, outputpath, chroma_method, dtwtype, dtwarea, verify_extremes, testpointsnum, diffpointstolerance, segmentdivider)
find_duplicates_img_hashing(path, outputpath, chroma_method, hashdiff_tresh)
find_duplicates_combined(path, outputpath, chroma_method, hashdiff_tresh, dtwtype, dtwarea, verify_extremes, testpointsnum, diffpointstolerance, segmentdivider)
is_chroma_duplicate(chroma1, chroma2, dtwtype, dtwarea, verify_extremes, testpointsnum, diffpointstolerance, segmentdivider, showplot)
is_duplicate(audiofile1_name, audiofile2_name, chroma_method, dtwtype, dtwarea, verify_extremes, testpointsnum, diffpointstolerance, segmentdivider, showplot)


------------------------ OUTPUT OF THE SYSTEM ------------------------------------
After finishing the calculation, the results can be found under the set outputpath directory (or in a DuplicateFinder folder under input path directory, if outputpath has not been set).
The system is creating output in two various formats - "DuplicateFinder_results", containing detailed information about every tested audio file, and "DuplicateFinder_results_duplicates_only", in which only duplicate pairs are recorded.
Both of these output formats are saved into .xlsx and .csv files for user to choose.

----------------------------------------------------------------------------------
Made as a thesis project by Bc. Michael Švejcar,
Brno University of Technology, 
Faculty of Electrical Engineering and Communication,
2022