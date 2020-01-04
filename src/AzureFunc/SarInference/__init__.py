import joblib
import json
import logging
from pathlib import Path

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('SarInference function processing request.')

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        tweet = req_body.get('tweet')

    try:
        p = Path(__file__).resolve().parents[0] / "model.joblib"
        model = joblib.load(p)
    except:
        return func.HttpResponse(
            f"No model found at {str(p)}",
            status_code=500
        )

    if tweet:
        logging.info('Tweet text: %s', tweet)

        try:
            res = do_inference()
        except:
            return func.HttpResponse(
                f"Error during inference.",
                status_code=500
            )

        return func.HttpResponse(
            body=json.dumps(res),
            status_code=200
        )
    else:
        return func.HttpResponse(
             "No tweet text passed in the request body.",
             status_code=400
        )
