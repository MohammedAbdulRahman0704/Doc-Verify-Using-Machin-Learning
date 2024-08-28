import os
import re
from flask import Flask, request, redirect, url_for, render_template, flash, send_file
from werkzeug.utils import secure_filename
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'docx', 'pdf', 'rtf', 'odt', 'html', 'xml', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # Set file size limit to 256 MB
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.root_path, 'instance', 'results.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'professor29'

socketio = SocketIO(app)
db = SQLAlchemy(app)

class PlagiarismResult(db.Model):
    """Model to store plagiarism check results."""
    id = db.Column(db.Integer, primary_key=True)
    first_file_name = db.Column(db.String(120), nullable=False)
    second_file_name = db.Column(db.String(120), nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    highlighted_result_path = db.Column(db.String(120), nullable=False)

def is_allowed_file(filename):
    """Check if the file has an allowed extension."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    """Sanitize the filename to prevent directory traversal attacks."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

# Create upload folder if it does not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create database tables
with app.app_context():
    db.create_all()

@app.before_request
def limit_file_size():
    """Check the size of the uploaded file."""
    if 'file' in request.files:
        file = request.files['file']
        if file and not is_allowed_file(file.filename):
            flash('Invalid file type.')
            return redirect(url_for('home'))

@app.route('/')
def home():
    """Render the home page with results from the database."""
    results = PlagiarismResult.query.all()
    return render_template('index.html', results=results)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and process them based on selected comparison type."""
    uploaded_files = [f for f in request.files.values() if is_allowed_file(f.filename)]
    if len(uploaded_files) < 2:
        flash('Please upload at least two files.')
        return redirect(url_for('home'))

    file_paths = [os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)) for f in uploaded_files]
    for f, path in zip(uploaded_files, file_paths):
        f.save(path)

    try:
        comparison_type = request.form.get('choice')  # Use 'choice' to get comparison type
        if comparison_type == 'similarity':
            comparison_results = check_plagiarism(file_paths)
        else:
            comparison_results = check_differences(file_paths)

        highlighted_files_info = []
        for result in comparison_results:
            if comparison_type == 'similarity':
                highlighted_path = highlight_similarities(file_paths[result[0] - 1], file_paths[result[1] - 1])
            else:
                highlighted_path = highlight_differences(file_paths[result[0] - 1], file_paths[result[1] - 1])

            highlighted_files_info.append({
                'highlighted_file': highlighted_path,
                'choice': comparison_type,
                'first_file_name': os.path.basename(file_paths[result[0] - 1]),
                'second_file_name': os.path.basename(file_paths[result[1] - 1]),
                'similarity_score': result[2]  # Assuming result[2] contains the similarity score
            })
            plagiarism_result = PlagiarismResult(
                first_file_name=os.path.basename(file_paths[result[0] - 1]),
                second_file_name=os.path.basename(file_paths[result[1] - 1]),
                similarity_score=result[2],
                highlighted_result_path=highlighted_path
            )
            db.session.add(plagiarism_result)
        db.session.commit()

        return render_template('index.html', results=highlighted_files_info)
    except FileNotFoundError as e:
        flash(f'File not found: {e.filename}')
        return redirect(url_for('home'))

def vectorize_text(text_list):
    """Convert a list of texts into TF-IDF vectors."""
    return TfidfVectorizer().fit_transform(text_list).toarray()

def calculate_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    return cosine_similarity([vector1, vector2])[0][1]

def check_plagiarism(file_paths):
    """Check plagiarism by comparing all file pairs."""
    file_contents = [open(fp, encoding='utf-8').read() for fp in file_paths]
    vectors = vectorize_text(file_contents)
    return [(i + 1, j + 1, calculate_similarity(vectors[i], vectors[j])) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]

def check_differences(file_paths):
    """Check differences between all file pairs."""
    file_contents = [open(fp, encoding='utf-8').read() for fp in file_paths]
    vectors = vectorize_text(file_contents)
    return [(i + 1, j + 1, 1 - calculate_similarity(vectors[i], vectors[j])) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]

def highlight_similarities(file_path1, file_path2):
    """Generate an HTML file highlighting similarities between two text files."""
    with open(file_path1, encoding='utf-8') as file1, open(file_path2, encoding='utf-8') as file2:
        text1 = file1.read()
        text2 = file2.read()

        matcher = SequenceMatcher(None, text1, text2)
        similarities = matcher.get_matching_blocks()

        highlighted_text1 = []
        highlighted_text2 = []

        last_a = last_b = 0
        for match in similarities:
            a, b, size = match
            if size > 0:
                highlighted_text1.append(text1[last_a:a])
                highlighted_text1.append(f'<span class="similar">{text1[a:a+size]}</span>')
                highlighted_text2.append(text2[last_b:b])
                highlighted_text2.append(f'<span class="similar">{text2[b:b+size]}</span>')
                last_a = a + size
                last_b = b + size

        highlighted_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.basename(file_path1)}_vs_{os.path.basename(file_path2)}_highlighted.html")
        with open(highlighted_file_path, 'w', encoding='utf-8') as highlighted_file:
            highlighted_file.write(f'''<html>
    <head>
        <style>
            body {{
                font-family: 'Roboto', sans-serif;
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: #ffffff;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}
            .container {{
                max-width: 800px;
                background: rgba(255, 255, 255, 0.1);
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            .container:hover {{
                transform: translateY(-10px);
                box-shadow: 0 16px 24px rgba(0, 0, 0, 0.5);
            }}
            h2 {{
                color: #00c6ff;
                text-align: center;
                margin-bottom: 30px;
            }}
            .file-container {{
                border: 1px solid rgba(255, 255, 255, 0.3);
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 10px;
                background-color: rgba(0, 0, 0, 0.5);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                transition: transform 0.3s ease, background-color 0.3s ease;
            }}
            .file-container:hover {{
                transform: translateY(-5px);
                background-color: rgba(0, 0, 0, 0.7);
            }}
            h3 {{
                color: #ff9800;
                margin-bottom: 10px;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
                color: #e0e0e0;
                background-color: #1e1e1e;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            .similar {{
                background-color: lightcoral;
                padding: 3px 5px;
                border-radius: 5px;
                font-weight: bold;
                transition: background-color 0.3s ease;
            }}
            .similar:hover {{
                background-color: #F08080;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Comparing {os.path.basename(file_path1)} and {os.path.basename(file_path2)}</h2>
            <div class="file-container">
                <h3>File 1:</h3>
                <pre>{''.join(highlighted_text1)}</pre>
            </div>
            <div class="file-container">
                <h3>File 2:</h3>
                <pre>{''.join(highlighted_text2)}</pre>
            </div>
        </div>
    </body>
    </html>''')

        return os.path.basename(highlighted_file_path)

def highlight_differences(file_path1, file_path2):
    """Generate an HTML file highlighting differences between two text files."""
    with open(file_path1, encoding='utf-8') as file1, open(file_path2, encoding='utf-8') as file2:
        text1 = file1.read()
        text2 = file2.read()

        matcher = SequenceMatcher(None, text1, text2)
        differences = matcher.get_opcodes()

        highlighted_text1 = []
        highlighted_text2 = []

        for tag, i1, i2, j1, j2 in differences:
            if tag == 'replace':
                highlighted_text1.append(f'<span class="difference">{text1[i1:i2]}</span>')
                highlighted_text2.append(f'<span class="difference">{text2[j1:j2]}</span>')
            elif tag == 'delete':
                highlighted_text1.append(f'<span class="difference">{text1[i1:i2]}</span>')
            elif tag == 'insert':
                highlighted_text2.append(f'<span class="difference">{text2[j1:j2]}</span>')
            elif tag == 'equal':
                highlighted_text1.append(text1[i1:i2])
                highlighted_text2.append(text2[j1:j2])

        highlighted_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.basename(file_path1)}_vs_{os.path.basename(file_path2)}_highlighted.html")
        with open(highlighted_file_path, 'w', encoding='utf-8') as highlighted_file:
            highlighted_file.write(f'''<html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            body {{
                font-family: 'Roboto', sans-serif;
                background: linear-gradient(135deg, #1e3c72, #2a5298);
                color: #ffffff;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }}
            .container {{
                max-width: 850px;
                background: rgba(255, 255, 255, 0.1);
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
                backdrop-filter: blur(15px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            .container:hover {{
                transform: translateY(-10px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
            }}
            h2 {{
                color: #00e6ff;
                text-align: center;
                margin-bottom: 35px;
            }}
            .file-container {{
                border: 1px solid rgba(255, 255, 255, 0.4);
                padding: 25px;
                margin-bottom: 25px;
                border-radius: 10px;
                background-color: rgba(0, 0, 0, 0.6);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
                transition: transform 0.3s ease, background-color 0.3s ease;
            }}
            .file-container:hover {{
                transform: translateY(-5px);
                background-color: rgba(0, 0, 0, 0.8);
            }}
            h3 {{
                color: #ffcc00;
                margin-bottom: 15px;
            }}
            pre {{
                white-space: pre-wrap;
                word-wrap: break-word;
                color: #f0f0f0;
                background-color: #272727;
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            .difference {{
                background-color: #00008B;
                color: #ffffff;
                padding: 3px 5px;
                border-radius: 5px;
                font-weight: bold;
                transition: background-color 0.3s ease;
            }}
            .difference:hover {{
                background-color: #00008B;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Comparing {os.path.basename(file_path1)} and {os.path.basename(file_path2)}</h2>
            <div class="file-container">
                <h3>File 1:</h3>
                <pre>{''.join(highlighted_text1)}</pre>
            </div>
            <div class="file-container">
                <h3>File 2:</h3>
                <pre>{''.join(highlighted_text2)}</pre>
            </div>
        </div>
    </body>
    </html>''')

        return os.path.basename(highlighted_file_path)


@app.route('/download_highlighted_file/<filename>')
def download_highlighted_file(filename):
    """Serve a highlighted file for download."""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/reset')
def reset():
    """Reset the application by removing all files and recreating the database."""
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
    db.drop_all()
    db.create_all()
    flash('All uploaded files and results have been reset.')
    return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

if __name__ == '__main__':
    socketio.run(app, debug=True)