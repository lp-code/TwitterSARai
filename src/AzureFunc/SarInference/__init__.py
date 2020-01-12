import joblib
import json
import logging
from pathlib import Path

import azure.functions as func

from .data_utils import split_into_tags_and_doc


VERSION = "2.0.0"


def logged_error_response(msg: str, status_code: int) -> func.HttpResponse:
    logging.error(msg)
    return func.HttpResponse(msg, status_code=status_code)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("SarInference function processing request.")

    try:
        req_body = req.get_json()
    except ValueError:
        tweet = None
    else:
        tweet = req_body.get("tweet")

    try:
        p = Path(__file__).resolve().parents[0] / "model.joblib"
        model = joblib.load(p)
    except Exception as e:
        return logged_error_response(
            f"No model found at {str(p)}: " + repr(e), status_code=500
        )

    if tweet:
        logging.info(f"Tweet text: {tweet}")

        try:
            tags, txt = split_into_tags_and_doc(tweet)
            res_array = model.predict([txt])
        except Exception as e:
            return logged_error_response(
                "Error during inference: " + repr(e), status_code=500
            )

        logging.info(f"Inference successful, result: class={res_array[0]}")
        return func.HttpResponse(
            body=json.dumps(
                dict(
                    tags=[] if not tags else tags.split("|"),
                    text=txt,
                    label=int(res_array[0]),
                    original=tweet,
                    version=VERSION,
                )
            ),
            mimetype="application/json",
            status_code=200,
        )
    else:
        return logged_error_response(
            "No tweet text passed in the request body.", status_code=400
        )
