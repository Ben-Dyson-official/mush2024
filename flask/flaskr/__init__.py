from flask import Flask, redirect, request
import os

def create_app():
	app = Flask(__name__)

	# Make instance path if not exists
	try:
		os.makedirs(app.instance_path)
	except OSError:
		pass

	from . import home
	app.register_blueprint(home.bp)

	return app

