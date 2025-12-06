from fastapi import FastAPI
from pydantic import BaseModel
from textsummarization.pipeline.training_pipeline import TrainingPipeline
from textsummarization.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Text Summarization API")

prediction_pipeline = PredictionPipeline()  # load model at startup


class TrainResponse(BaseModel):
    message: str


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictResponse(BaseModel):
    summary: str


class PredictBatchResponse(BaseModel):
    summaries: list[str]


@app.post("/train", response_model=TrainResponse)
def train_pipeline():
    pipeline = TrainingPipeline()
    pipeline.run()
    return TrainResponse(message="Training completed successfully")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    summary = prediction_pipeline.predict(req.text)
    return PredictResponse(summary=summary)


@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    summaries = prediction_pipeline.predict_batch(req.texts)
    return PredictBatchResponse(summaries=summaries)
