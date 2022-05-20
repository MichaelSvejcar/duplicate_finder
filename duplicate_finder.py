# -*- coding: utf-8 -*-

import os
import sys
import csv
import openpyxl
from openpyxl import load_workbook, workbook
from contextlib import contextmanager
import librosa
import numpy as np
from matplotlib import pyplot as plt
from nnAudio import features
import time
import os.path
from PIL import Image
import imagehash
from dtw import dtw
from scipy.spatial.distance import euclidean
from synctoolbox.dtw import mrmsdtw
from fastdtw import fastdtw
import torch
import numpy as np
import subprocess as sp
from csv import writer
import os
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS

DEVNULL = open(os.devnull, 'w')

# Global settings variables -------------------------------------------------------------------------------------
fs = 11025
output_filename = "DuplicateFinder_results"
output_filename_duplicates_only = "DuplicateFinder_results_duplicates_only"
# Global variables for functions --------------------------------------------------------------------------------
spec_layer = None

# Functions for audio duplicates finding ------------------------------------------------------------------------
def find_duplicates(path, outputpath = None, accuracy = "normal", chroma_method= "cuda"):
    """
    Explanation: Simplest function to run duplicate finding algorithm using four different accuracy settings
    :param path: Directory of folder in which to look for duplicates
    :param outputpath: Output directory for chroma features and final results; if not set, folder gets automatically created inside the path folder
    :param accuracy: Calculation accuracy, can be set to "low", "normal", "high" and "extreme"; "low" for finding duplicates which are exactly identical, "normal" for cases where there may be some sort of noise in the beggining/end of one duplicate or for case when one duplicate is encoded into very low bitrate, "high" is similar as normal, but has even lower tolerance for differences, "extreme" can be used for cases when user expects very long passages (like half of the whole recording) of noise in beggining/end of some of the audio duplicates
    :param chroma_method: Method for chroma features calculation, "cuda" is for nnAudio implementation with the use of CUDA, "synctoolbox" uses functions from same-named library without CUDA - dependency
    """
    if accuracy == "low":
        find_duplicates_img_hashing(path, outputpath, chroma_method, hashdiff_tresh = 0)
    elif accuracy == "normal":
        find_duplicates_combined(path, outputpath, chroma_method, hashdiff_tresh = 10, dtwarea = 1000000, verify_extremes = False)
    elif accuracy == "high":
        find_duplicates_combined(path, outputpath, chroma_method, hashdiff_tresh = 20, dtwarea = 1000000, verify_extremes = False)
    elif accuracy == "extreme":
        find_duplicates_dtw(path, outputpath, dtwarea = 10000000, verify_extremes = True)

def find_duplicates_dtw(path, outputpath = None, chroma_method = "cuda", dtwtype ="mrmsdtw", dtwarea = 1000000, verify_extremes = False, testpointsnum = 100, diffpointstolerance = 5, segmentdivider = 4):
    """
    Explanation: Function that iterates through user-defined audio files directory to find duplicates using DTW method, writes output to .csv and .xlsx files
    :param path: Directory of folder in which to look for duplicates
    :param outputpath: Output directory for chroma features and final results; if not set, folder gets automatically created inside the path folder
    :param chroma_method: Method for chroma features calculation, "cuda" is for nnAudio implementation with the use of CUDA, "synctoolbox" uses functions from same-named library without CUDA - dependency
    :param dtwtype: Type of used DTW method (can be: mrmsdtw, dtw, fastdtw)
    :param dtwarea: For setting unique DTW method parameter, depends on used dtwtype - for mrmsdtw this parameter sets tau, using fastdtw it is defining radius
    :param verify_extremes: Sets whether path evaluation is done for both orientations of the axis; set to true, if you expect very long passages of silence, applause etc. in beginning of one of the recordings
    :param testpointsnum: number of points tested between referential points
    :param diffpointstolerance: determines percentage of tested points whose value may be different, for which the system still evaluates the recordings as the same
    :param segmentdivider: determines from which points is the referential line counted - for example, if segmentdivider = 4, testing line will start from 1/4 and end in 3/4 range of the warping path
    """

    filedirs_all = np.asarray(librosa.util.find_files(path, ext=['mp3', 'mp4', 'ogg', 'wav'])).tolist() # list containing paths to all audiofiles (every subdirectory)

    if (outputpath == None): # if outputpath is not defined, it gets created inside path directory
        outputpath = os.path.join(path, "DuplicateFinder")
    if (os.path.isdir(outputpath) == False): # create data output directory if doesnt exist yet
        os.mkdir(outputpath)
    csvfiledir = os.path.join(outputpath, output_filename + ".csv")
    excelfiledir = os.path.join(outputpath, output_filename + ".xlsx")
    csvfiledir_duplicates_only = os.path.join(outputpath, output_filename_duplicates_only + ".csv")
    excelfiledir_duplicates_only = os.path.join(outputpath, output_filename_duplicates_only + ".xlsx")

    # starts the time counter
    tcalcstart = time.time()

    # chromagrams calculation
    _calculate_chromagrams(path, outputpath, chroma_method)

    # DTW calculation (returns list of duplicates)
    duplicatepairslist = _return_duplicates_dtw(path, outputpath, chroma_method, dtwtype, dtwarea, verify_extremes, testpointsnum, diffpointstolerance, segmentdivider)

    # writes to output files
    _create_output_files(duplicatepairslist, filedirs_all, csvfiledir, excelfiledir, csvfiledir_duplicates_only, excelfiledir_duplicates_only)

    # stops the timer and writes output to console
    tcalcend = time.time()
    calctime = round(tcalcend - tcalcstart, 2)

    numofduplicatepairs = len(duplicatepairslist)

    print("\nCalculation finished!")
    print("Total calculation time: " + str(calctime) + " s")
    print("Number of duplicate pairs found: " + str(numofduplicatepairs))

def find_duplicates_img_hashing(path, outputpath = None, chroma_method = "cuda", hashdiff_tresh = 10):
    """
    Explanation: Function that iterates through user-defined audio files directory to find duplicates using image hashing method, writes output to .csv and .xlsx files
    :param path: Directory of folder in which to look for duplicates
    :param outputpath: Output directory for chroma features and final results; if not set, folder gets automatically created inside the path folder
    :param method: "cuda" for chroma features calculation using CUDA and nnAudio, "synctoolbox" for using same-named library
    :param hashdiff_tresh: Treshold of hash difference, for which two recordings are evaluated as same
    """
    filedirs_all = np.asarray(librosa.util.find_files(path, ext=['mp3', 'mp4', 'ogg', 'wav'])).tolist() # list containing paths to all audiofiles (every subdirectory)
    
    if (outputpath == None): # if outputpath is not defined, it gets created inside path directory
        outputpath = os.path.join(path, "DuplicateFinder")
    if (os.path.isdir(outputpath) == False): # create data output directory if doesnt exist yet
            os.mkdir(outputpath)
    csvfiledir = os.path.join(outputpath, output_filename + ".csv")
    excelfiledir = os.path.join(outputpath, output_filename + ".xlsx")
    csvfiledir_duplicates_only = os.path.join(outputpath, output_filename_duplicates_only + ".csv")
    excelfiledir_duplicates_only = os.path.join(outputpath, output_filename_duplicates_only + ".xlsx")

    chroma_folder = os.path.join(outputpath, "chroma_files")
    chroma_imgs_folder = os.path.join(outputpath, "chroma_files_imgs")

    # starts the time counter
    tcalcstart = time.time()

    # chroma features calculation
    _calculate_chromagrams(path, outputpath, chroma_method)

    # exports chroma features as images
    _export_chromafiles_as_imgs(chroma_folder, chroma_imgs_folder)

    # duplicates calculation
    pairslist = _return_duplicates_img_hashing(path, outputpath, hashdiff_tresh)

    # stops the timer and writes output to console
    tcalcend = time.time()
    calctime = round(tcalcend - tcalcstart, 2)

    numofduplicatepairs = len(pairslist)

    print("\nCalculation finished!")
    print("Total calculation time: " + str(calctime) + " s")
    print("Number of duplicate pairs found: " + str(numofduplicatepairs))

    # writing results to output file
    _create_output_files(pairslist, filedirs_all, csvfiledir, excelfiledir, csvfiledir_duplicates_only, excelfiledir_duplicates_only)

def find_duplicates_combined(path, outputpath = None, chroma_method = "cuda", hashdiff_tresh = 10, dtwtype ="mrmsdtw", dtwarea = 1000000, verify_extremes = False, testpointsnum = 100, diffpointstolerance = 5, segmentdivider = 4):
    """
    Explanation: Function that iterates through user-defined audio files directory to find duplicates using image hashing first to check which pairs might be similar, and then evaluating these found pairs using DTW method
    :param path: Directory of folder in which to look for duplicates
    :param outputpath: Output directory for chroma features and final results; if not set, folder gets automatically created inside the path folder
    :param chroma_method: Method for chroma features calculation, "cuda" is for nnAudio implementation with the use of CUDA, "synctoolbox" uses functions from same-named library without CUDA - dependency
    :param hashdiff_tresh: Treshold of hash difference, for which two recordings are evaluated as same
    :param dtwtype: Type of used DTW method (can be: mrmsdtw, dtw, fastdtw)
    :param dtwarea: For setting unique DTW method parameter, depends on used dtwtype - for mrmsdtw this parameter sets tau, using fastdtw it is defining radius
    :param verify_extremes: Sets whether path evaluation is done for both orientations of the axis; set to true, if you expect very long passages of silence, applause etc. in beginning of one of the recordings
    :param testpointsnum: number of points tested between referential points
    :param diffpointstolerance: determines percentage of tested points whose value may be different, for which the system still evaluates the recordings as the same
    :param segmentdivider: determines from which points is the referential line counted - for example, if segmentdivider = 4, testing line will start from 1/4 and end in 3/4 range of the warping path
    """
    
    filedirs_all = np.asarray(librosa.util.find_files(path, ext=['mp3', 'mp4', 'ogg', 'wav'])).tolist() # list containing paths to all audiofiles (every subdirectory)

    if (outputpath == None): # if outputpath is not defined, it gets created inside path directory
        outputpath = os.path.join(path, "DuplicateFinder")
    if (os.path.isdir(outputpath) == False): # create data output directory if doesnt exist yet
        os.mkdir(outputpath)
    csvfiledir = os.path.join(outputpath, output_filename + ".csv")
    excelfiledir = os.path.join(outputpath, output_filename + ".xlsx")
    csvfiledir_duplicates_only = os.path.join(outputpath, output_filename_duplicates_only + ".csv")
    excelfiledir_duplicates_only = os.path.join(outputpath, output_filename_duplicates_only + ".xlsx")

    chroma_folder = os.path.join(outputpath, "chroma_files")
    chroma_imgs_folder = os.path.join(outputpath, "chroma_files_imgs")

    # starts the time counter
    tcalcstart = time.time()

    # chromagrams calculation
    _calculate_chromagrams(path, outputpath, chroma_method)

    # exports chroma features as images
    _export_chromafiles_as_imgs(chroma_folder, chroma_imgs_folder)

    # image hashing duplicates pre-calculation
    pairslistimghashing = _return_duplicates_img_hashing(path, outputpath, hashdiff_tresh)

    # DTW calculation of only pairs pre-calculated by image hashing
    pairslistfinal = _return_duplicates_dtw(path, outputpath, chroma_method, dtwtype, dtwarea, verify_extremes, testpointsnum, diffpointstolerance, segmentdivider, pairslistimghashing)

    # writes to output files
    _create_output_files(pairslistfinal, filedirs_all, csvfiledir, excelfiledir, csvfiledir_duplicates_only, excelfiledir_duplicates_only)

    # stops the timer and writes output to console
    tcalcend = time.time()
    calctime = round(tcalcend - tcalcstart, 2)

    numofduplicatepairs = len(pairslistfinal)

    print("\nCalculation finished!")
    print("Total calculation time: " + str(calctime) + " s")
    print("Number of duplicate pairs found: " + str(numofduplicatepairs))

def is_chroma_duplicate(chroma1, chroma2, dtwtype = "mrmsdtw", dtwarea = 1000000, verify_extremes = False, testpointsnum = 100, diffpointstolerance = 5, segmentdivider = 4, showplot=False):
    """
    Explanation: Checks if two chromagrams corresponding to recordings are duplicates or not
    :param chroma1: Chromagram of first recording
    :param chroma2: Chromagram of second recording
    :param dtwtype: Type of used DTW method (can be: mrmsdtw, dtw, fastdtw)
    :param dtwarea: For setting unique DTW method parameter, depends on used dtwtype - for mrmsdtw this parameter sets tau, using fastdtw it is defining radius
    :param verify_extremes: Sets whether path evaluation is done for both orientations of the axis; set to true, if you expect very long passages of silence, applause etc. in beginning of one of the recordings
    :param testpointsnum: number of points tested between referential points
    :param diffpointstolerance: determines percentage of tested points whose value may be different, for which the system still evaluates the recordings as the same
    :param segmentdivider: determines from which points is the referential line counted - for example, if segmentdivider = 4, testing line will start from 1/4 and end in 3/4 range of the warping path
    :param showplot: if set to true, function will plot the results
    :return: returns true if two input chromagrams are the same, returns false otherwise
    """

    # DTW ---------------------------
    if dtwtype == "mrmsdtw":
        path = mrmsdtw.sync_via_mrmsdtw(chroma1, chroma2, dtw_implementation="librosa", threshold_rec=dtwarea)
        pathx = np.array(path[0,:]) # rozdeleni cesty do dvou np arrays
        pathy = np.array(path[1,:])
    elif dtwtype == "fastdtw":
        # flipnuti os chroma vektoru
        chroma1 = np.swapaxes(chroma1, 0, 1)
        chroma2 = np.swapaxes(chroma2, 0, 1)
        distance, path = fastdtw(x = chroma1, y = chroma2, dist = euclidean, radius = dtwarea)
        pathx, pathy = zip(*path[::-1]) # reverse osy aby slo vzestupne a rozdeleni do dvou samostatnych arrays
        pathx = np.array(pathx) # prevedeni na datovy typ array
        pathy = np.array(pathy)
    elif dtwtype == "dtw":
        chroma1 = np.swapaxes(chroma1, 0, 1)
        chroma2 = np.swapaxes(chroma2, 0, 1)
        path = dtw(chroma1, chroma2, dist = euclidean)
        pathx = path[3][0] # rozdeleni cesty do dvou np arrays
        pathy = path[3][1]
    else:
        print("Wrong dtwtype input argument!")
        quit()

    issame, plt = _verify_path_flatness(pathx, pathy, testpointsnum = testpointsnum, diffpointstolerance = diffpointstolerance, segmentdivider = segmentdivider)
    if (verify_extremes): 
        if (issame == False): # if the system returns that files are not duplicates, it flips the axes and verifies in this order aswell (which can help if there is for example very long passage of noise in one of the audio files)
            plt.clf()
            issame, plt = _verify_path_flatness(pathy, pathx, testpointsnum = testpointsnum, diffpointstolerance = diffpointstolerance, segmentdivider = segmentdivider)

    if (showplot == True):
        plt.show()

    return issame

def is_duplicate(audiofile1_name, audiofile2_name, chroma_method = "cuda", dtwtype = "mrmsdtw", dtwarea = 1000000, verify_extremes = False, testpointsnum = 100, diffpointstolerance = 5, segmentdivider = 4, showplot=False):
    """
    Explanation: Checks if two audio files are duplicates or not
    :param audiofile1_name: Path to first audio file
    :param audiofile2_name: Path to second audio file
    :param chroma_method: Method for chroma features calculation, "cuda" is for nnAudio implementation with the use of CUDA, "synctoolbox" uses functions from same-named library without CUDA - dependency
    :param dtwtype: Type of used DTW method (can be: mrmsdtw, dtw, fastdtw)
    :param dtwarea: For setting unique DTW method parameter, depends on used dtwtype - for mrmsdtw this parameter sets tau, using fastdtw it is defining radius
    :param testpointsnum: number of points tested between referential points
    :param diffpointstolerance: determines percentage of tested points whose value may be different, for which the system still evaluates the recordings as the same
    :param segmentdivider: determines from which points is the referential line counted - for example, if segmentdivider = 4, testing line will start from 1/4 and end in 3/4 range of the warping path
    :param showplot: if set to true, function will plot the results
    :return: returns true if two input audio files are the same, returns false otherwise
    """

    audio1, _ = _ffmpeg_load_audio(audiofile1_name, sr = fs, mono = True)
    audio2, _ = _ffmpeg_load_audio(audiofile2_name, sr = fs, mono = True)

    if (chroma_method == "cuda"):
        chroma1 = _calculate_chromagram_cuda(audio1)
        chroma2 = _calculate_chromagram_cuda(audio2)
    elif (chroma_method == "synctoolbox"):
        chroma1 = _calculate_chromagram_synctoolbox(audio1)
        chroma2 = _calculate_chromagram_synctoolbox(audio2)

    print("Checking whether files \"" + os.path.basename(audiofile1_name) + "\" and \"" + os.path.basename(audiofile2_name) + "\" are duplicates:")
    isDuplicate = is_chroma_duplicate(chroma1, chroma2, dtwtype, dtwarea, verify_extremes, testpointsnum, diffpointstolerance, segmentdivider, showplot)
    print(isDuplicate)

    return isDuplicate

# Helper functions ----------------------------------------------------------------------------------------------

# function for rewriting cells in .csv
def _csvaddtocell(csvdir, row, column, value):
    f = open(csvdir, 'r', encoding = "utf-8")
    reader = csv.reader(f)
    mylist = list(reader)
    f.close()

    if(len(mylist[row][column]) == 0):
        mylist[row][column] = str(value)
    else:
        mylist[row][column] = str(mylist[row][column]) + ", " + str(value)

    mylistnew = open(csvdir, 'w', newline='', encoding="utf-8")
    csv_writer = csv.writer(mylistnew)
    csv_writer.writerows(mylist)
    mylistnew.close()

# function that appends list to .csv
def _append_row_csv(csvdir, list):
    with open(csvdir, 'a', newline='') as f_object:  
        # Pass the CSV  file object to the writer() function
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow(list)  
        # Close the file object
        f_object.close()

# function that exports csv data to xlsx
def convert_csv_to_xlsx(csvfile, xlsxfile):
    wb = openpyxl.Workbook()
    ws = wb.active
    with open(csvfile, 'r', encoding = "utf-8") as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(xlsxfile)

# function for audio file loading using FFMPEG
def _ffmpeg_load_audio(filename, sr=44100, mono=False, normalize=True, in_type=np.int16, out_type=np.float32):
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    p = sp.Popen(command, stdout=sp.PIPE, stderr=DEVNULL, bufsize=4096)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.fromstring(raw, dtype=in_type).astype(out_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    if audio.size == 0:
        return audio, sr
    if issubclass(out_type, np.floating):
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio /= peak
        elif issubclass(in_type, np.integer):
            audio /= np.iinfo(in_type).max
    return audio, sr

# function for output files creating
def _create_output_files(duplicatepairslist, filedirs_all, csvfiledir, excelfiledir, csvfiledir_duplicates_only, excelfiledir_duplicates_only):
    numofduplicatepairs = len(duplicatepairslist)

    if os.path.exists(csvfiledir_duplicates_only):
        os.remove(csvfiledir_duplicates_only)
    # writes header to csv file containing only list of duplicates
    _append_row_csv(csvfiledir_duplicates_only, ["File 1 directory", "File 1 name", "File 2 directory", "File 2 name"])

    # finds i and j coordinates from duplicatepairslist containing all the files
    for duplicatepair in duplicatepairslist:
        file1name = os.path.basename(duplicatepair[0]) # returns only filename with extension
        file2name = os.path.basename(duplicatepair[1])

        i = filedirs_all.index(duplicatepair[0])
        j = filedirs_all.index(duplicatepair[1])

        # writes to csv file 
        _csvaddtocell(csvfiledir, i+1, 3, file2name)
        _csvaddtocell(csvfiledir, i+1, 4, j)
        _csvaddtocell(csvfiledir, j+1, 3, file1name)
        _csvaddtocell(csvfiledir, j+1, 4, i)

        # writes to csv file duplicates only
        _append_row_csv(csvfiledir_duplicates_only, [duplicatepair[0], file1name, duplicatepair[1], file2name])

    # converts csvs to excel files
    convert_csv_to_xlsx(csvfiledir, excelfiledir)
    convert_csv_to_xlsx(csvfiledir_duplicates_only, excelfiledir_duplicates_only)

# function that returns nearest value to the input value from array
def _find_nearest(array, value): 
    index = np.abs(array - value).argmin()
    return array.flat[index]


# chroma features calculation functions --------------------------------------------------------------------------

# function for chroma features calculation using CUDA and nnAudio
def _calculate_chromagram_cuda(audio):
    # initializes spectrogram layer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    global spec_layer
    if (spec_layer == None):
        if (torch.cuda.is_available()):
            spec_layer = features.CQT(sr=fs, hop_length=512).cuda()
        else:
            spec_layer = features.CQT(sr=fs, hop_length=512).cpu()


    # creates cqt spektrogramu using nnaudio, to parse into librosa
    audio = torch.tensor(audio, device=device).float()  # casting the array into a PyTorch Tensor
    cqt = spec_layer(audio)
    cqt = cqt.cpu().detach().numpy()[0]

    # calculates chromagram
    chroma = librosa.feature.chroma_cqt(C=cqt, sr=fs, hop_length=512)
    return chroma

# function for chroma features calculation using synctoolbox
def _calculate_chromagram_synctoolbox(audio):
    f_pitch = audio_to_pitch_features(audio, Fs = fs)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    
    return f_chroma_quantized

# function for chroma features calculation of all files from defined path
def _calculate_chromagrams(path, outputpath, chroma_method):
    # loading of directory with subfolders
    audiofolderlist = [] # list only for folders with audio files (not containing chroma_files or DuplicateChecker)
    subfolders = [x[0] for x in os.walk(path)]
    for folder in subfolders:
        if not ( "chroma_files" in folder or "DuplicateFinder" in folder): # only folders which are not for duplicatechecker data
                if not (np.asarray(librosa.util.find_files(folder, ext=['mp3', 'mp4', 'ogg', 'wav'], recurse = False)).size == 0): # only folders containing audio files
                    audiofolderlist.append(folder) # appends 

    filedirs = np.asarray(librosa.util.find_files(path, ext=['mp3', 'mp4', 'ogg', 'wav'])) # list containing paths to all audiofiles (every subdirectory)
    filesnumber = filedirs.size

    
    # Initializes csv writer and output dir
    csvfiledir = os.path.join(outputpath, output_filename + ".csv")
    header = ['File ID', 'File directory', 'File name', 'Duplicate file names', 'Duplicate IDs']
    csvfile = open(csvfiledir, 'w', encoding="utf-8", newline='')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    
    currentfilenum = 0
    # iterates through folders with audio data
    for folder in audiofolderlist:
        folderreldir = os.path.relpath(folder, path)
        chromapath = os.path.join(outputpath, "chroma_files", folderreldir)

        # Creates chroma output directory if it doesnt exist yet
        if (os.path.isdir(chromapath) == False):
            os.makedirs(chromapath)

        foldercurrentfilenum = 0 # variable for indexing, unique for each subdirectory
        folderfiledirs = np.asarray(librosa.util.find_files(folder, ext=['mp3', 'mp4', 'ogg', 'wav'], recurse = False)) # list containing audio file paths+names in current subfolder

        excelrow = 1
        for audiofile in folderfiledirs:
            filename = os.path.basename(audiofile)
            filenamewithext = filename + ".npy"
            chromafilepath = os.path.join(chromapath, filenamewithext)
            filedir = folderfiledirs[foldercurrentfilenum]

            # Calculates chroma features of audio file if it hasnt been calculated yet (doesnt exist in chroma_files folder)
            if (os.path.exists(chromafilepath) == False): 
                print("Extracting chroma features from file \"" + os.path.basename(audiofile) + "\" (" + str(currentfilenum+1) + "/" + str(filesnumber) + ")")
                wave, _ = _ffmpeg_load_audio(audiofile, sr=fs, mono=True)

                if (chroma_method == "cuda"):
                    chroma = _calculate_chromagram_cuda(wave)
                elif (chroma_method == "synctoolbox"):
                    chroma = _calculate_chromagram_synctoolbox(wave)
                    print("\n")

                np.save(chromafilepath, chroma)
            else:
                print("Chroma features corresponding to file: \"" + os.path.basename(audiofile) + "\" have been loaded!" + " (" + str(currentfilenum+1) + "/" + str(filesnumber) + ")")

            csvwriter.writerow([currentfilenum, filedir, filename, "", ""])

            foldercurrentfilenum = foldercurrentfilenum + 1
            currentfilenum = currentfilenum + 1

    csvfile.close()
    print("\nChroma features have been successfuly extracted from all audio files!\n")

# function that evaluates DTW path flatness (whether two recordings are same or not)
def _verify_path_flatness(pathx, pathy, testpointsnum, diffpointstolerance, segmentdivider):
    # determination of sample numbers for line approximation
    pathxminval = min(pathx) # determination of min and max values (start and beginning of line on x axis)
    pathxmaxval = max(pathx)
    pathxvalrange = pathxmaxval - pathxminval

    # makes sure that the range is divisible by the segmentdivider value
    modulo = pathxvalrange % segmentdivider
    pathxvalrange = pathxvalrange - modulo

    refpoint1xval = int(pathxminval+(pathxvalrange/segmentdivider)) # determination of point x value for approximation
    refpoint2xval = int(pathxminval+(pathxvalrange/segmentdivider)*(segmentdivider-1))
    refpoint1xpos = int(np.argwhere(pathx==refpoint1xval)[0]) # finds out positions of array pathx at which these points are located
    refpoint2xpos = int(np.argwhere(pathx==refpoint2xval)[0])

    refpointsx = np.array([pathx[refpoint1xpos], pathx[refpoint2xpos]]) # creates arrays with x and y coordinates in format suitable for np.polyfit
    refpointsy = np.array([pathy[refpoint1xpos], pathy[refpoint2xpos]])

    # Line approximation --------------------------------
    coefficients = np.polyfit(refpointsx, refpointsy, 1)
    polynomial = np.poly1d(coefficients)

    linex = np.arange(start=0, stop=len(pathx), step=1)
    liney = polynomial(linex)

    # verifies whether the path between the two points used to approximate the curve actually lies on the curve
    refpointsvaldiff = refpoint2xval - refpoint1xval

    if (testpointsnum > refpointsvaldiff): # ensures that the number of test points does not exceed the number of defined points between the reference points except the reference points themselves
        testpointsnum = refpointsvaldiff - 1

    testpointstep = refpointsvaldiff / testpointsnum # step size

    # Testing of points ---------------------------------
    testpointxshift = testpointstep/2 # variable that ensures that the first test point is not at the point where the curve intersects path
    diffpointsnum = 0 # a variable to which one is added in the cycle iteration if the point values ​​do not fit
    for i in range(0, testpointsnum, 1): # cycle iterating across individual testing points
        testpointxval = refpoint1xval + i*testpointstep + testpointxshift # finding the value for testing

        testpointnearestxval = _find_nearest(pathx, testpointxval) # finding the nearest value to the test value
        testpointxpos = np.argwhere(pathx==testpointnearestxval)[0] # finding the first index of the path element, which is equal to the given test position (index of pathx does not always match the value !!)
        pathval = float(pathy[testpointxpos]) # finding the value that matches the index found in the previous step
        lineval = float(liney[int(testpointnearestxval)]) # finding the value of the tested position at the approximation curve (here the index is always equal to the value)



        if (abs(pathval-lineval)>1.2): # if the difference between the values ​​is greater than 1.2, it gets written to diffpointsnum
            diffpointsnum = diffpointsnum+1
            plt.plot(testpointnearestxval, pathval, '+', markersize=12, color='red')
        else:
            plt.plot(testpointnearestxval, pathval, '+', markersize=12, color='green')


    diffpointsnumtolerance = round((diffpointstolerance/100) * testpointsnum)
    # Evaluation, if two recordings are the same ---------
    if (diffpointsnum <= diffpointsnumtolerance):
        issame = True
    else:
        issame = False

    # PLOTTING
    plt.plot(pathx, pathy, color="black")
    plt.plot(linex[refpoint1xval:refpoint2xval+1], liney[refpoint1xval:refpoint2xval+1])
    plt.plot(refpointsx, refpointsy, 'o', markersize=8, color='blue')

    return issame, plt

# chromagram to image functions ----------------------------------------------------------------------------------
def _load_chromafile_and_save_as_img(chroma_filename, output_filename):
    chroma = np.load(chroma_filename)
    chromaprocessed = (chroma * 256).astype(np.uint8) # get into right scale
    im = Image.fromarray(chromaprocessed)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(output_filename) # save

# function that takes folder of chromagrams as input and exports them as bitmap into defined output folder
def _export_chromafiles_as_imgs(chroma_folder, output_folder):
    # creates directory if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # lists all chroma files in all subdirectories of chroma_folder
    chromalist = list()
    for (dirpath, dirnames, filenames) in os.walk(chroma_folder):
        chromalist += [os.path.join(dirpath, file) for file in filenames]

    print("Saving chroma features as bitmaps for image hashing")

    for file in chromalist:
        chromadir = os.path.realpath(file) #input chroma dir
        chromareldir = os.path.relpath(chromadir, chroma_folder)
        chromaimgreldir = chromareldir.replace('.npy', '') + ".png"
        chromaoutputdir = os.path.join(output_folder, chromaimgreldir)

        # creates directory if it doesnt exist yet
        directory = os.path.dirname(chromaoutputdir)
        if os.path.isdir(directory) == False:
            os.makedirs(directory)

        if (os.path.exists(chromaoutputdir) == False):
            _load_chromafile_and_save_as_img(chromadir, chromaoutputdir)

# function that evaluates, whether two images are similar according to set hash difference treshold
def _are_imgs_similar(file1, file2, hashdiff_tresh):
    hash1 = imagehash.phash(Image.open(file1))
    hash2 = imagehash.phash(Image.open(file2))

    hashdiff = abs(hash1 - hash2)
    if (hashdiff <= hashdiff_tresh):
        return True
    else:
        return False


# duplicate finding helper functions -----------------------------------------------------------------------------

# function that returns list of found duplicates in defined path using DTW method
def _return_duplicates_dtw(path, outputpath, chroma_method = "cuda", dtwtype ="mrmsdtw", dtwarea = 1000000, verify_extremes = False, testpointsnum = 100, diffpointstolerance = 5, segmentdivider = 4, filepairslist = None):

    filenames=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(tuple(['mp3', 'mp4', 'ogg', 'wav'])):
                filenames.append(file)

    filedirs_all = np.asarray(librosa.util.find_files(path, ext=['mp3', 'mp4', 'ogg', 'wav'])) # list containing paths to all audiofiles (every subdirectory)
    filesnumber = filedirs_all.size
    
    currentpairnum = 1

    # if filepairslist to test is not defined from function argument, it gets set to all possible combinations
    if (filepairslist == None):
        filepairslist = []

        for i in range(0, filesnumber):
            for j in range(i+1, filesnumber):
                filepairslist.append([filedirs_all[i], filedirs_all[j]])

    numofpairs = len(filepairslist)
    filedirs_all = filedirs_all.tolist()
    duplicatepairslist = []

    #Checking every pair if it is a duplicate
    for filepair in filepairslist:
        file1dir = str(filepair[0])
        file2dir = str(filepair[1])
        file1reldir = os.path.dirname(os.path.relpath(file1dir, path)) # returns relative directory to file (removes c:\\ and path (project folder))
        file2reldir = os.path.dirname(os.path.relpath(file2dir, path))

        file1name = os.path.basename(filepair[0]) # returns only filename with extension
        file2name = os.path.basename(filepair[1])
        file1namenpy = file1name + ".npy"
        file2namenpy = file2name + ".npy"

        chroma1dir = os.path.join(outputpath, "chroma_files", file1reldir, file1namenpy)
        chroma2dir = os.path.join(outputpath, "chroma_files", file2reldir, file2namenpy)

        print("Using DTW to check whether files \"" + file1name + "\" and \"" + file2name + "\" are duplicates" + " (" + str(currentpairnum) + "/" + str(numofpairs) + ")")

        chroma1 = np.load(chroma1dir)
        chroma2 = np.load(chroma2dir)

        isduplicate = is_chroma_duplicate(chroma1, chroma2, dtwtype, dtwarea, verify_extremes, testpointsnum, diffpointstolerance, segmentdivider)
        print(isduplicate)

        if (isduplicate):
            duplicatepairslist.append([file1dir, file2dir])
        currentpairnum = currentpairnum + 1

    return duplicatepairslist

# function that returns list of found duplicates in defined path using image hashing
def _return_duplicates_img_hashing(path, outputpath, hashdiff_tresh = 10):
    filenames=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(tuple(['mp3', 'mp4', 'ogg', 'wav'])):
                filenames.append(file)

    filedirs_all = np.asarray(librosa.util.find_files(path, ext=['mp3', 'mp4', 'ogg', 'wav'])) # list containing paths to all audiofiles (every subdirectory)
    filesnumber = filedirs_all.size
    
    currentpairnum = 1

    filepairslist = []
    for i in range(0, filesnumber):
        for j in range(i+1, filesnumber):
            filepairslist.append([filedirs_all[i], filedirs_all[j]])

    numofpairs = len(filepairslist)
    filedirs_all = filedirs_all.tolist()
    duplicatepairslist = []

    #Checking every pair if it is a duplicate
    for filepair in filepairslist:
        file1dir = str(filepair[0])
        file2dir = str(filepair[1])
        file1reldir = os.path.dirname(os.path.relpath(file1dir, path)) # returns relative directory to file (removes c:\\ and path (project folder))
        file2reldir = os.path.dirname(os.path.relpath(file2dir, path))

        file1name = os.path.basename(filepair[0]) # returns only filename with extension
        file2name = os.path.basename(filepair[1])

        file1namepng = file1name + ".png"
        file2namepng = file2name + ".png"

        chromaimg1dir = os.path.join(outputpath, "chroma_files_imgs", file1reldir, file1namepng)
        chromaimg2dir = os.path.join(outputpath, "chroma_files_imgs", file2reldir, file2namepng)

        print("Using image hashing to check whether files \"" + file1name + "\" and \"" + file2name + "\" are similar" + " (" + str(currentpairnum) + "/" + str(numofpairs) + ")")

        isduplicate = _are_imgs_similar(chromaimg1dir, chromaimg2dir, hashdiff_tresh)
        print(isduplicate)

        if (isduplicate):
            duplicatepairslist.append([file1dir, file2dir])
        currentpairnum = currentpairnum + 1

    return duplicatepairslist

