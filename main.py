#!flask/bin/python
import os
import sys
from PIL import Image
import json
from flask import Flask, jsonify, abort, make_response, request, redirect, url_for, send_from_directory, flash, session
from flask_session import Session
from collections import defaultdict
from werkzeug.utils import secure_filename
from werkzeug import SharedDataMiddleware
from inspect import getmembers
from pprint import pprint
from shapedetector import ShapeDetector
import argparse
import imutils
import cv2
from googletrans import Translator

app = Flask(__name__)
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'images/')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
sess = Session()
application = ClarifaiApp(api_key='eb7f93f1cce54defb92e276ab515d948')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_ext(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower()

app.add_url_rule('/images/<filename>', 'uploaded_file', build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/images':  app.config['UPLOAD_FOLDER']
})

@app.route('/images/predict', methods=['GET','POST'])
def get_prediction():
    output = defaultdict(list)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file_to_be_saved' not in request.files:
            output['status'] = 'false'
            output['message'] = 'error request'
        else:
            file = request.files['file_to_be_saved']
            # if user does not select file, browser also submit a empty part without filename
            if file.filename == '':
                output['status'] = 'false'
                output['message'] = 'no file selected'
            else:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(save_path)
                    img = Image.open(save_path)
                    base = int(float(img.size[0])/2.5)
                    wpercent = (base / float(img.size[0]))
                    hsize = int((float(img.size[1]) * float(wpercent)))
                    img = img.resize((base, hsize), Image.ANTIALIAS)
                    img.save(save_path, dpi=[100,100])
                    dump_image = cv2.imread(save_path)
                     
                    # convert the resized image to grayscale, blur it slightly,
                    # and threshold it
                    gray = cv2.cvtColor(dump_image, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
                     
                    # find contours in the thresholded image and initialize the
                    # shape detector
                    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                    sd = ShapeDetector()

                    # loop over the contours
                    shape_feature = {}
                    shape_feature['abstrak'] = 0
                    shape_feature['kotak'] = 0
                    shape_feature['persegi'] = 0
                    shape_feature['lingkaran'] = 0
                    shape_feature['segitiga'] = 0
                    shape_feature['segilima'] = 0
                    shape_feature['segienam'] = 0
                    for c in cnts:
                        shape = sd.detect(c)
                        
                        if(shape == 'abstrak'):
                            shape_feature['abstrak'] += 1
                        elif(shape=='kotak'):
                            shape_feature['kotak'] += 1
                        elif(shape=='persegi'):
                            shape_feature['persegi'] += 1
                        elif(shape=='lingkaran'):
                            shape_feature['lingkaran'] += 1
                        elif(shape=='segitiga'):
                            shape_feature['segitiga'] += 1
                        elif(shape=='segilima'):
                            shape_feature['segilima'] += 1
                        else:
                            shape_feature['segienam'] += 1

                    workflow = application.workflows.get('custom-models')
                    if os.path.exists(save_path):
                        open_image = open(save_path,'rb')
                        image = ClImage(file_obj = open_image)
                        translator = Translator()
                        result = []
                        result = json.loads(json.dumps(workflow.predict([image])))
                        open_image.close()
                        if result['status']['code'] == 10000:
                            i = 0
                            for data in result['results']:                        
                                if data['status']['code'] == 10000:
                                    output_feature = defaultdict(list)
                                    index = 0
                                    output_feature['url'] = url_for('uploaded_file',filename=filename)
                                    output_feature['image'] = filename
                                    output_feature['shapes'] = shape_feature
                                    for segment in data['outputs']:
                                        if index == 0:
                                            if len(segment['data']['concepts']) > 0:
                                                for item in segment['data']['concepts']:
                                                    feature = {}
                                                    translation = translator.translate(item['name'],dest='id')
                                                    feature['kata'] = translation.text
                                                    feature['score'] = item['value']
                                                    output_feature['features'].append(feature)
                                        if index == 1:
                                            if len(segment['data']['colors']) > 0:
                                                for item in segment['data']['colors']:
                                                    feature = {}
                                                    translation = translator.translate(item['w3c']['name'],dest='id')
                                                    feature['kode'] = item['w3c']['hex']
                                                    h = feature['kode'].lstrip('#')
                                                    rgb = tuple(int(h[j:j+2], 16) for j in (0, 2 ,4))
                                                    rgb_schem = {}
                                                    rgb_schem['r'] = rgb[0]
                                                    rgb_schem['g'] = rgb[1]
                                                    rgb_schem['b'] = rgb[2]
                                                    feature['rgb'] = rgb_schem
                                                    feature['warna'] = translation.text
                                                    feature['score'] = item['value']
                                                    output_feature['colors'].append(feature)
                                        if index == 2:
                                            if len(segment['data']['concepts']) > 0:
                                                for item in segment['data']['concepts']:
                                                    feature = {}
                                                    translation = translator.translate(item['name'],dest='id')
                                                    feature['label'] = translation.text
                                                    feature['score'] = item['value']
                                                    output_feature['textures'].append(feature)
                                        index = index+1
                                    output[i] = output_feature
                                    i = i+1
                    else:
                        output['status'] = 'false'
                        output['message'] = 'file not saved'
                else:
                    output['status'] = 'false'
                    output['message'] = 'file not accepted'
    else:
        output['status'] = 'false'
        output['message'] = 'error request'
    return jsonify(output)
    # return '''
    # <!doctype html>
    # <title>Upload new File</title>
    # <h1>Upload new File</h1>
    # <form method=post enctype=multipart/form-data>
    #   <p><input type=file name=file>
    #      <input type=submit value=Upload>
    # </form>
    # '''

@app.route('/images/softpredict', methods=['POST'])
def get_softprediction():
    import glob
    output = defaultdict(list)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file_to_be_saved' not in request.files:
            output['status'] = 'false'
            output['message'] = 'error request'
        else:
            file = request.files['file_to_be_saved']
            if file.filename == '':
                output['status'] = 'false'
                output['message'] = 'no file selected'
            else:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    saved_image = os.listdir(os.path.join(app.config['UPLOAD_FOLDER']))
                    if filename in saved_image:
                        filename = 'dump_'+filename
                        filename = secure_filename(filename)
                    else:
                        filename = filename
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(save_path)
                    img = Image.open(save_path)
                    base = int(float(img.size[0])/2.5)
                    wpercent = (base / float(img.size[0]))
                    hsize = int((float(img.size[1]) * float(wpercent)))
                    img = img.resize((base, hsize), Image.ANTIALIAS)
                    img.save(save_path, dpi=[100,100])
                    dump_image = cv2.imread(save_path)
                     
                    # convert the resized image to grayscale, blur it slightly,
                    # and threshold it
                    gray = cv2.cvtColor(dump_image, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
                     
                    # find contours in the thresholded image and initialize the
                    # shape detector
                    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                    sd = ShapeDetector()

                    # loop over the contours
                    shape_feature = {}
                    shape_feature['abstrak'] = 0
                    shape_feature['kotak'] = 0
                    shape_feature['persegi'] = 0
                    shape_feature['lingkaran'] = 0
                    shape_feature['segitiga'] = 0
                    shape_feature['segilima'] = 0
                    shape_feature['segienam'] = 0
                    for c in cnts:
                        shape = sd.detect(c)
                        
                        if(shape == 'abstrak'):
                            shape_feature['abstrak'] += 1
                        elif(shape=='kotak'):
                            shape_feature['kotak'] += 1
                        elif(shape=='persegi'):
                            shape_feature['persegi'] += 1
                        elif(shape=='lingkaran'):
                            shape_feature['lingkaran'] += 1
                        elif(shape=='segitiga'):
                            shape_feature['segitiga'] += 1
                        elif(shape=='segilima'):
                            shape_feature['segilima'] += 1
                        else:
                            shape_feature['segienam'] += 1

                    workflow = application.workflows.get('custom-models')
                    if os.path.exists(save_path):
                        open_image = open(save_path,'rb')
                        image = ClImage(file_obj = open_image)
                        
                        translator = Translator()
                        result = []
                        result = json.loads(json.dumps(workflow.predict([image])))
                        if filename not in saved_image:
                            open_image.close()
                            os.unlink(save_path)
                        if result['status']['code'] == 10000:
                            i = 0
                            for data in result['results']:                        
                                if data['status']['code'] == 10000:
                                    output_feature = defaultdict(list)
                                    index = 0
                                    output_feature['url'] = url_for('uploaded_file',filename=filename)
                                    output_feature['image'] = filename
                                    output_feature['shape'] = shape_feature
                                    for segment in data['outputs']:
                                        if index == 0:
                                            if len(segment['data']['concepts']) > 0:
                                                for item in segment['data']['concepts']:
                                                    feature = {}
                                                    translation = translator.translate(item['name'],dest='id')
                                                    feature['kata'] = translation.text
                                                    feature['score'] = item['value']
                                                    output_feature['description'].append(feature)
                                        if index == 1:
                                            if len(segment['data']['colors']) > 0:
                                                for item in segment['data']['colors']:
                                                    feature = {}
                                                    translation = translator.translate(item['w3c']['name'],dest='id')
                                                    feature['kode'] = item['w3c']['hex']
                                                    h = feature['kode'].lstrip('#')
                                                    rgb = tuple(int(h[j:j+2], 16) for j in (0, 2 ,4))
                                                    rgb_schem = {}
                                                    rgb_schem['r'] = rgb[0]
                                                    rgb_schem['g'] = rgb[1]
                                                    rgb_schem['b'] = rgb[2]
                                                    feature['rgb'] = rgb_schem
                                                    feature['warna'] = translation.text
                                                    feature['score'] = item['value']
                                                    output_feature['color'].append(feature)
                                        if index == 2:
                                            if len(segment['data']['concepts']) > 0:
                                                for item in segment['data']['concepts']:
                                                    feature = {}
                                                    translation = translator.translate(item['name'],dest='id')
                                                    feature['label'] = translation.text
                                                    feature['score'] = item['value']
                                                    output_feature['texture'].append(feature)
                                        index = index+1
                                    output[i] = output_feature
                                    i = i+1
                    else:
                        output['status'] = 'false'
                        output['message'] = 'file not saved'
                else:
                    output['status'] = 'false'
                    output['message'] = 'file not accepted'
    else:
        output['status'] = 'false'
        output['message'] = 'error request'
    return jsonify(output)

@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.secret_key = 'I02UASNOACUONACCAjksancuebondcoanoianPNOICANS'
    app.config['SESSION_TYPE'] = 'filesystem'

    sess.init_app(app)
    app.run(debug=True)