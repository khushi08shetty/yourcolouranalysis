from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import base64
import numpy as np
from your_color_analysis import analyze_image  # Import your color analysis function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:  # For file uploads
        file = request.files['image']
        if file.filename == '':
            return redirect(url_for('index'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
    elif 'image_data' in request.form:  # For captured photos
        image_data = request.form['image_data']
        header, encoded = image_data.split(',', 1)
        image_data = base64.b64decode(encoded)
        filename = 'captured_image.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Convert to OpenCV image and save
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(filepath, image)
    else:
        return redirect(url_for('index'))

    # Pass the image to your color analysis function
    analysis_result = analyze_image(filepath)
    return render_template('result.html', result=analysis_result, image_path=filename)

if __name__ == '__main__':
    app.run(debug=True)
