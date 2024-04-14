from flask import Blueprint, render_template, redirect, request, flash
import os
import random
import csv
import util

bp = Blueprint("home", __name__, url_prefix="/")

# routes

def read_csv(path):
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        facts = list(reader)

        random_fact = random.choice(facts[1:])

    return random_fact

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
    fact = read_csv('../constellation_facts.csv')

    return render_template("home/home.html", filename=file.filename, fact=fact)
