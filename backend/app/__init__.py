"""
MiroFish Backend - Flask application factory
"""

import os
import warnings

# Suppress multiprocessing resource_tracker warnings (from third-party libraries such as transformers)
# Must be set before all other imports
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask application factory function"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Configure JSON encoding: ensure non-ASCII characters are displayed directly (instead of \uXXXX format)
    # Flask >= 2.3 uses app.json.ensure_ascii; older versions use the JSON_AS_ASCII config
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # Set up logging
    logger = setup_logger('mirofish')
    
    # Only print startup info in the reloader child process (avoid printing twice in debug mode)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend starting...")
        logger.info("=" * 50)
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register simulation process cleanup function (ensure all simulation processes are terminated when the server shuts down)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Registered simulation process cleanup function")
    
    # Request logging middleware
    @app.before_request
    def reject_path_like_identifiers():
        # Identifiers become folder names, so "../" could reach outside the data folder.
        values = list((request.view_args or {}).values())
        values += [v for k, v in request.args.items() if k.endswith('_id')]
        body = request.get_json(silent=True) if request.is_json else None
        if isinstance(body, dict):
            values += [v for k, v in body.items() if k.endswith('_id')]
        for value in values:
            if isinstance(value, str) and (
                '..' in value or '/' in value or chr(92) in value or chr(0) in value
            ):
                return jsonify({"success": False, "error": "Invalid identifier"}), 400

    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"Request: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"Request body: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"Response: {response.status_code}")
        return response
    
    # Register blueprints
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    
    # Health check
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}
    
    if should_log_startup:
        logger.info("MiroFish Backend startup complete")
    
    return app

