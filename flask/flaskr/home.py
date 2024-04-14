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
	option = request.form.get('checkbox-slider')
	
	file = request.files['inputFile']
	file.save(os.path.join('./flaskr/static/', file.filename))

	if option: #Constellation option
		index = util.check_model(os.path.join('./flaskr/static/', file.filename))[0]
		fact = util.read_csv(index+1)
		return render_template("home/home.html", filename=file.filename, fact=fact)
	else: #Star option
		star_num, cluster_num = util.classify(file.filename)
		return render_template("home/home.html", filename=file.filename, star_num=star_num, cluster_num=cluster_num)



