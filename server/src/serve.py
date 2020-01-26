import numpy as np
from flask import Flask, abort, jsonify, render_template, request
from pythonosc import dispatcher, osc_message_builder, osc_server, udp_client

from src.generator import Generator
from src.model import CVAE, VAE, VGG16VAE

app = Flask(__name__)

with open("config.json", "r") as f:
    conf = json.load(f)

if conf["use-osc"]:
    address = "127.0.0.1"
    client = udp_client.UDPClient(address, conf["osc-port"])

G = Generator(VAE(), "models/hoge.ckpt")


@app.route('/api/generate', methods=['POST'])
def generate():
    """

    """
    z: list = request.json.z
    images: list = G.generate(np.array(z), base64=True)
    data = {
        "base64": images
    }
    return jsonify(data)
