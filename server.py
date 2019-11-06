from flask import Flask, request,render_template, jsonify
from werkzeug import secure_filename
from keras.models import load_model
import pandas as pd
import uuid
import json
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump, load
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
import os
import re
import numpy as np
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
from sklearn.externals.joblib import dump, load





app = Flask(__name__)
model = load_model('vgg16.h5')
scaler=load('std_scaler.bin')



def contrast_feature(matrix_coocurrence):
	contrast = greycoprops(matrix_coocurrence, 'contrast')
	return contrast

def dissimilarity_feature(matrix_coocurrence):
	dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
	return dissimilarity

def homogeneity_feature(matrix_coocurrence):
	homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
	return homogeneity

def energy_feature(matrix_coocurrence):
	energy = greycoprops(matrix_coocurrence, 'energy')
	return energy

def correlation_feature(matrix_coocurrence):
	correlation = greycoprops(matrix_coocurrence, 'correlation')
	return correlation

def asm_feature(matrix_coocurrence):
	asm = greycoprops(matrix_coocurrence, 'ASM')
	return asm


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/tumor_detection', methods = ['POST'])
def tumor_detection():
	df = pd.DataFrame(columns = [val for val in range(32)])

	if request.method == 'POST':
		idx = 0
		filedata = request.files['imgfile']
		file_ext = filedata.filename.split(".")[-1]
		pid = uuid.uuid4()
		unique_filename = str(pid) + '.' + file_ext
		file_path = os.path.join('static/imgdata',unique_filename)
		filedata.save(file_path)


		unique_filename_output = str(pid) + "_output" + "." + file_ext
		file_path_output = os.path.join('static/imgdata', unique_filename_output)

		img = np.array(Image.open(file_path))
		img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		img_bgr = cv2.resize(img_bgr,None,(256,256))
		# bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
		# inds = np.digitize(gray, bins)
		# max_value = inds.max()+1
		# matrix_coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=True, symmetric=True)
		# a = contrast_feature(matrix_coocurrence)
		# b = dissimilarity_feature(matrix_coocurrence)
		# c = homogeneity_feature(matrix_coocurrence)
		# d = energy_feature(matrix_coocurrence)
		# e = correlation_feature(matrix_coocurrence)
		# f = asm_feature(matrix_coocurrence)
		# g = a+f
		# h = a+e+d+c
		# x = np.concatenate([a,b,c,d,e,f,g,h],axis=1)[0]
		# df.loc[idx]=x
		#
		# df = scaler.transform(df)

		model_output = model.predict(img_bgr)
		model_output = int(model_output)
		print(model_output)

		api_output = {'status': 'Success', 'data': model_output, 'file_url_original':file_path, 'file_url_output': file_path_output}
		return jsonify(api_output)
	else:
		api_output = {'status': 'Invalid request type'}
		return jsonify(api_output)

if __name__=='__main__':
    app.run(host='0.0.0.0')
