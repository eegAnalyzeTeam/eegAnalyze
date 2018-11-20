# This program is where the main function at. 
# The program calls preprocessing functions, calculate psds for each
# person and use anova test to find the significant channels and sub-frequencies
import mne
import numpy as np
from matplotlib import pyplot as plt
import os, sys
import check_file

def readData(filePath):
	#q = ['njh_after_pjk_20180725_close.vhdr', 'ccs_yb_20180813_close.vhdr']
	q = ['njh_after_pjk_20180725_close.vhdr', 'ccs_yb_20180813_close.vhdr', 'njh_before_pjk_20180613_close.vhdr', 'ccs_before_wjy_20180817_close.vhdr', 'ccs_after_csx_20180511_close.vhdr']

	control_raw = {}
	control_q = []
	patient_raw = {}
	patient_q = []
	for dirpath, dirs, files in os.walk(filePath):
		# print('dirpath: ' + dirpath)
		# print('dirs: --------------------' )
		# print(dirs)
		# print('files: ==================')
		# print(files)
		if 'eyeclose' in dirpath and 'health_control' in dirpath:
			#health control group
			for fname in files:
				if '.vhdr' in fname:
					print('control+'+fname)
					id_control = fname[:-5]
					vmrkf,eegf = check_file.get_vhdr_info(dirpath + '/' + fname) 
					if vmrkf == eegf and vmrkf == id_control:
						print('OK')
					else:
						print('control: vhdr:' + id_control + ' vmrk: ' + vmrkf + ' eeg:' + eegf)
						control_q.append(id_control)

					raw = mne.io.read_raw_brainvision(dirpath + '/' + fname,preload=True)

					raw.set_montage(mne.channels.read_montage("standard_1020"))
					control_raw[id_control] = raw

		elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
			#mdd group
			for fname in files:
				if '.vhdr' in fname:
					if fname in q:
						continue
					print('patient+'+fname)
					id_patient = fname[:-5]
					vmrkf,eegf = check_file.get_vhdr_info(dirpath + '/' + fname) 
					if vmrkf == eegf and vmrkf == id_patient:
						print('OK')
					else:
						print('patient: vhdr:' + id_patient + ' vmrk: ' + vmrkf + ' eeg:' + eegf)
						patient_q.append(id_patient)

					raw = mne.io.read_raw_brainvision(dirpath + '/' + fname,preload=True)

					raw.set_montage(mne.channels.read_montage("standard_1020"))
					patient_raw[id_patient] = raw

	return control_raw, patient_raw
	#return control_q, patient_q
#raw = mne.io.read_raw_brainvision('/home/caeit/Documents/work/eeg/eegData/mdd_patient/eyeopen/njh_after_pjk_20180725_open.vhdr',preload=True)
#raw = mne.io.read_raw_brainvision('/home/caeit/Documents/work/eeg/eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',preload=True)


control_raw, patient_raw = readData('/home/caeit/Documents/work/eeg/eegData')
#control_q, patient_q = readData('/home/caeit/Documents/work/eeg/eegData')
print(control_raw)
print('=================')
print(patient_raw)
print('control: ' + str(len(control_raw)))
print('patient: ' + str(len(patient_raw)))
