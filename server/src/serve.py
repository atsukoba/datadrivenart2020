import json
import numpy as np
from flask import Flask, abort, jsonify, render_template, request
from flask_cors import CORS
from pythonosc import dispatcher, osc_message_builder, osc_server, udp_client

from src.generator import Generator
from src.model import CVAE, VAE, VGG16VAE

app = Flask(__name__)
CORS(app)


with open("config.json", "r") as f:
    conf = json.load(f)

if conf["use-osc"]:
    address = "127.0.0.1"
    client = udp_client.UDPClient(address, conf["osc-port"])

G = Generator(VAE(), "models/sample_albumcover_vae_64_10epoch.torch")


@app.route('/api/generate', methods=['POST'])
def generate():
    """get JSON data of generated images"""
    z: list = request.json["z"]
    print(z[0])
    images: list = G.generate(z, use_base64=True)
    data = {
        "base64": images
    }
    print(data)
    return jsonify(data)


@app.route('/api/generate', methods=['GET'])
def help():
    """API check"""
    return jsonify({"command": "POST JSON data ({file: [0.1, 0.2, 0.4,...,0.6]}) to /api/generate"})
