from flask import Blueprint, render_template, redirect, request, flash
import os

import util

bp = Blueprint("home", __name__, url_prefix="/")

# routes

@bp.route("")
def handle_empty():
    return redirect("/home")

@bp.route("home")
def home_page():
	return render_template("home/home.html")

@bp.route("home", methods=['POST'])
def image_submit():
	file = request.files['inputFile']
	file.save(os.path.join('./flaskr/static/', file.filename))
	util.classify(file.filename)
	return render_template("home/home.html", filename=file.filename)
