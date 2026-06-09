"""from it.combibrivioSLR.flask.flask_manager import FlaskManager

app = FlaskManager()
app.run(host='0.0.0.0', port=5000, debug = True)

"""

from it.combibrivioSLR.dataset.dataset_manager import DatasetManager
#from it.combibrivioSLR.flask.flask_manager import FlaskManager
#from it.combibrivioSLR.machine_learning.regressione_logistica import RegLogistica
#from it.combibrivioSLR.machine_learning.random_forest import RndForest


#app = FlaskManager()
#app.run(host='0.0.0.0', port=5000, debug=True)


print("Carico dataset")
ds_mg = DatasetManager()
print()

print("stampa dataset")
ds_mg.stampa()
print()

print("Esplorazione del dataset")
ds_mg.analisi()
print()

print("Visualizzazione grafici")
#ds_mg.grafici()
print()

print("pulisco dataset")
ds_mg.clean()
print()

print()
ds_mg.analisi()


print("stampa dataset dopo cleaning")
ds_mg.stampa()
print()

print("correlazione")
#print(ds_mg.correlazione())
print()



