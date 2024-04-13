from flask import Blueprint, render_template, redirect, request, flash
import os

import util

bp = Blueprint("home", __name__, url_prefix="/")

# routes

@bp.route("")
def handleEmpty():
    return redirect("/home")

@bp.route("home")
def homePage():
	return render_template("home/home.html")

@bp.route("home", methods=['POST'])
def imageSubmit():
	file = request.files['inputFile']
	file.save(os.path.join('./', file.filename))	
	util.classify(file.filename)
	return redirect("/home")


