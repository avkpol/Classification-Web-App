import os
from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from flask_bootstrap import Bootstrap
import tensorflow as tf
import classifier


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
bootstrap = Bootstrap(app)

STATIC_FOLDER = "static"
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads/images/")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cnn_model = tf.keras.models.load_model(
    os.path.join(STATIC_FOLDER, "uploads/models/cats_dogs_trained_model.keras")
)


class UploadForm(FlaskForm):
    image = FileField('Upload an image', validators=[FileRequired()])


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        image = form.image.data
        filename = image.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)
        return redirect(url_for('results', filename=filename))

    return render_template('index.html', form=form, filename=None)


@app.route('/results/<filename>')
def results(filename):
    upload_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    label, prob = classifier.classify(cnn_model, upload_image_path)
    prob = round((prob * 100), 2)
    return render_template('results.html', label=label, prob=prob, filename=filename)


if __name__ == '__main__':
    app.run()
