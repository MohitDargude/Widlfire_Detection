from flask import Flask
import os

def create_app():
    # Absolute path to the templates folder
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Path to `app/`
    template_dir = os.path.join(base_dir, 'templates')     # Path to `app/templates`
    
    # Debugging: Confirm paths
    print(f"[DEBUG] Templates path: {template_dir}")
    print(f"[DEBUG] Index.html exists: {os.path.exists(os.path.join(template_dir, 'index.html'))}")
    
    # Initialize Flask
    app = Flask(__name__, template_folder=template_dir)
    
    # Configure upload folder
    app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, '..', 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register routes
    from .routes import main_routes
    app.register_blueprint(main_routes)
    
    return app