from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch


class ImageEncoder:
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def encode(self, batch):
        with torch.no_grad():
            pixel_values = self.processor(batch, return_tensors="pt").pixel_values
            emb = self.model.encoder(pixel_values).last_hidden_state
            return emb.mean(dim=1)
