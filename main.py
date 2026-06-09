from it.combibrivioSLR.flask.flask_manager import FlaskManager

app = FlaskManager()
app.run(host='0.0.0.0', port=5000, debug = True)