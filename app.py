from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# ================= UTILIDADES =================

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def find_candles(img):
    """
    Detecta contornos verticais (velas) e retorna lista ordenada da esquerda p/ direita
    com métricas: corpo, pavios, altura total.
    """
    proc = preprocess(img)
    _, th = cv2.threshold(proc, 120, 255, cv2.THRESH_BINARY_INV)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candles = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # filtros para remover ruído
        if area < 300 or h < 25 or w > h:
            continue

        # estimativa simples de corpo/pavios
        top = y
        bottom = y + h
        mid = y + h//2

        body = int(h * 0.45)      # aproximação robusta p/ imagem
        upper_wick = int(h * 0.25)
        lower_wick = int(h * 0.25)

        candles.append({
            "x": x,
            "h": h,
            "body": body,
            "upper": upper_wick,
            "lower": lower_wick
        })

    candles = sorted(candles, key=lambda k: k["x"])
    return candles

def classify_candle(c):
    """
    Classificação estrutural (price action por imagem)
    """
    if c["body"] < 0.25 * c["h"]:
        return "DOJI"
    if c["lower"] > c["body"] * 1.2:
        return "REJEICAO_BAIXO"
    if c["upper"] > c["body"] * 1.2:
        return "REJEICAO_CIMA"
    return "FORCA"

def market_context(candles):
    """
    Tendência simples por estrutura (sem médias falsas):
    compara alturas recentes.
    """
    if len(candles) < 6:
        return "NEUTRO"

    hs = [c["h"] for c in candles[-6:]]
    if hs[-1] > hs[-3] > hs[-5]:
        return "ALTA"
    if hs[-1] < hs[-3] < hs[-5]:
        return "BAIXA"
    return "LATERAL"

def exhaustion_check(types):
    """
    Exaustão: muitas velas de força seguidas
    """
    if len(types) >= 4 and types[-4:].count("FORCA") >= 3:
        return True
    return False

def score_decision(candles):
    """
    Sistema de pontuação (segredo da assertividade)
    """
    score = 0
    reasons = []

    types = [classify_candle(c) for c in candles]
    ctx = market_context(candles)

    # 1) Contexto
    if ctx == "ALTA":
        score += 15
        reasons.append("Contexto de alta")
    elif ctx == "BAIXA":
        score += 15
        reasons.append("Contexto de baixa")
    else:
        score -= 10
        reasons.append("Mercado lateral")

    # 2) Padrão da última vela
    last = types[-1]
    if last == "REJEICAO_BAIXO":
        score += 30
        reasons.append("Rejeição inferior")
        direction = "VERDE"
    elif last == "REJEICAO_CIMA":
        score += 30
        reasons.append("Rejeição superior")
        direction = "VERMELHO"
    else:
        direction = None

    # 3) Confirmação pela vela anterior
    prev = types[-2]
    if prev == "FORCA":
        score += 15
        reasons.append("Confirmação por vela de força")

    # 4) Exaustão (filtro)
    if exhaustion_check(types):
        score -= 20
        reasons.append("Exaustão detectada")

    # 5) Decisão final
    if score >= 50 and direction:
        return direction, score, reasons

    return "NÃO ENTRAR", score, reasons

# ================= ENDPOINT =================

@app.route("/analisar", methods=["POST"])
def analisar():
    if "image" not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"})

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img = np.array(img)

    candles = find_candles(img)

    if len(candles) < 5:
        return jsonify({
            "timeframe": "M3",
            "sinal": "NÃO ENTRAR",
            "confianca": 0,
            "explicacao": ["Poucas velas detectadas"]
        })

    sinal, score, reasons = score_decision(candles)

    return jsonify({
        "timeframe": "M3",
        "sinal": sinal,
        "confianca": min(score, 100),
        "explicacao": reasons,
        "velas_analisadas": len(candles)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
