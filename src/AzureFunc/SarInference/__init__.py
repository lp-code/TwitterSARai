import joblib
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import azure.functions as func

from .data_utils import split_into_tags_and_doc


VERSION = "20200222.0"
THRESHOLD_DEFAULT = 0.03

def logged_error_response(msg: str, status_code: int) -> func.HttpResponse:
    logging.error(msg)
    return func.HttpResponse(msg, status_code=status_code, mimetype="text/plain")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("SarInference function processing request.")

    try:
        req_body = req.get_json()
    except ValueError:
        tweet = None
    else:
        tweet = req_body.get("tweet")

    try:
        # Have to add the current directory to the path so that the modules
        # copied to the current directory during deployment are found.
        current_dir = Path(__file__).resolve().parents[0]
        sys.path.append(str(current_dir))

        p = current_dir / "model.joblib"
        model = joblib.load(p)
    except Exception as e:
        return logged_error_response(
            f"No model found at {str(p)}: " + repr(e), status_code=500
        )

    if not tweet:
        return logged_error_response(
            "No tweet text passed in the request body.", status_code=400
        )
    else:
        logging.info(f"Tweet text: {tweet}")

    try:
        threshold = float(os.environ["AZTWITTERSARAI_THRESHOLD"])
        logging.info(f"Using threshold from env: {threshold}")
    except (KeyError, ValueError):
        threshold = THRESHOLD_DEFAULT
        logging.info(f"No valid threshold set in env, using default: {threshold}")

    try:
        tags, txt = split_into_tags_and_doc(tweet)
        res_array = model.predict_proba([txt])
    except Exception as e:
        return logged_error_response(
            "Error during inference: " + repr(e) + traceback.format_exc(), status_code=500
        )

    logging.info(f"Inference successful, result: class={res_array[0]}")
    return func.HttpResponse(
        body=json.dumps(
            dict(
                tags=[] if not tags else tags.split("|"),
                text=txt,
                label=int(res_array[0, 1] >= threshold),
                score=res_array[0, 1],
                original=tweet,
                version=VERSION,
            )
        ),
        mimetype="application/json",
        status_code=200,
    )
