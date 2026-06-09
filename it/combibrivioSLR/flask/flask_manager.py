from io import BytesIO
import base64

import pandas as pd
from flask import Flask, jsonify, request, render_template_string
import matplotlib.pyplot as plt

from it.combibrivioSLR.dataset.dataset_manager import DatasetManager
from it.combibrivioSLR.machine_learning.regressione_logistica import RegLogistica
from it.combibrivioSLR.machine_learning.random_forest import RndForest


class FlaskManager(object):
    def __init__(self):
        self.app = Flask(__name__)
        self.__register_routes()

        self.ds_mg = DatasetManager()
        self.ds_mg.clean()
        # print(self.ds_mg.get_datatset().loc[5].to_dict())

        self.reg_log = RegLogistica(self.ds_mg.get_datatset())
        self.rnd_forest = RndForest(self.ds_mg.get_datatset())

    def run(self, **kwargs):
        self.app.run(**kwargs)

    def __register_routes(self):
        @self.app.route('/')
        def home():
            return "Flask online"

        @self.app.route('/datasetshow')
        def dataset_show():
            return jsonify(self.ds_mg.get_datatset().head(5).to_dict())

        @self.app.route('/info')
        def info():
            risp = self.ds_mg.analisi()

            def convert(obj):
                if isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj

            risp = {k: convert(v) for k, v in risp.items()}
            return jsonify(risp)

        @self.app.route('/grafici')
        def grafici():
            grafici = self.ds_mg.grafici()
            figs = []
            for value in grafici.values():
                if value is None:
                    continue

                # Se è una lista di figure
                if isinstance(value, list):
                    figs.extend(value)
                else:
                    figs.append(value)

            images = []

            for fig in figs:
                img = BytesIO()
                fig.savefig(img, format="png", bbox_inches="tight")
                img.seek(0)

                encoded = base64.b64encode(img.getvalue()).decode()
                images.append(encoded)

                plt.close(fig)

            html = """
               <h1>Plots</h1>
               {% for img in images %}
                   <img src="data:image/png;base64,{{ img }}" style="margin:10px;">
               {% endfor %}
               """

            return render_template_string(html, images=images)

        @self.app.route('/correlazione')
        def correlazione():
            return jsonify(self.ds_mg.correlazione())

        @self.app.route('/valMod_regLogistica')
        def valMod_regLogistica():
            return jsonify(self.reg_log.get_val())

        @self.app.route('/valMod_rndForest')
        def valMod_regLogisticaRndForest():
            return jsonify(self.rnd_forest.get_val())

        @self.app.route('/previsione_regLogistica', methods=['POST'])
        def previsione_regLogistica():
            data = request.get_json()
            obj = [
                ('cap-shape', data.get('cap-shape')),
                ('cap-surface', data.get('cap-surface')),
                ('cap-color', data.get('cap-color')),
                ('bruises', data.get('bruises')),
                ('odor', data.get('odor')),
                ('gill-attachment', data.get('gill-attachment')),
                ('gill-spacing', data.get('gill-spacing')),
                ('gill-size', data.get('gill-size')),
                ('gill-color', data.get('gill-color')),
                ('stalk-shape', data.get('stalk-shape')),
                ('stalk-root', data.get('stalk-root')),
                ('stalk-surface-above-ring', data.get('stalk-surface-above-ring')),
                ('stalk-surface-below-ring', data.get('stalk-surface-below-ring')),
                ('stalk-color-above-ring', data.get('stalk-color-above-ring')),
                ('stalk-color-below-ring', data.get('stalk-color-below-ring')),
                ('veil-color', data.get('veil-color')),
                ('ring-number', data.get('ring-number')),
                ('ring-type', data.get('ring-type')),
                ('spore-print-color', data.get('spore-print-color')),
                ('population', data.get('population')),
                ('habitat', data.get('habitat'))
            ]

            pred = self.reg_log.prevedi(obj)
            print("PREDIZIONE:", pred)

            return jsonify({"poisonus": self.reg_log.prevedi(obj).tolist()})

        @self.app.route('/previsione_rndForest', methods=['POST'])
        def previsione_rndForest():
            data = request.get_json()
            obj = [
                ('cap-shape', data.get('cap-shape')),
                ('cap-surface', data.get('cap-surface')),
                ('cap-color', data.get('cap-color')),
                ('bruises', data.get('bruises')),
                ('odor', data.get('odor')),
                ('gill-attachment', data.get('gill-attachment')),
                ('gill-spacing', data.get('gill-spacing')),
                ('gill-size', data.get('gill-size')),
                ('gill-color', data.get('gill-color')),
                ('stalk-shape', data.get('stalk-shape')),
                ('stalk-root', data.get('stalk-root')),
                ('stalk-surface-above-ring', data.get('stalk-surface-above-ring')),
                ('stalk-surface-below-ring', data.get('stalk-surface-below-ring')),
                ('stalk-color-above-ring', data.get('stalk-color-above-ring')),
                ('stalk-color-below-ring', data.get('stalk-color-below-ring')),
                ('veil-color', data.get('veil-color')),
                ('ring-number', data.get('ring-number')),
                ('ring-type', data.get('ring-type')),
                ('spore-print-color', data.get('spore-print-color')),
                ('population', data.get('population')),
                ('habitat', data.get('habitat'))
            ]
            return jsonify({"poisonus": self.rnd_forest.prevedi(obj).tolist()})