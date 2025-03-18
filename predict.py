# Prediction interface for Cog ⚙️
# https://cog.run/python

import warnings
import numpy as np
from pathlib import Path
import onnxruntime
from cog import BasePredictor, Input, Path as CogPath
from PIL import Image
from torchvision import transforms


def softmax(x) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


class Predictor(BasePredictor):
    def setup(self):
        """Load the ONNX model into memory."""
        self.angles = [0, 90, 180, 270]  # classes
        self.model_path = "resnet152_ixion_e3-fac493d9.onnx"

        # Define providers, try CUDA first if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # Load ONNX model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                self.ort_session = onnxruntime.InferenceSession(
                    self.model_path,
                    providers=providers
                )
                print(f"ONNX model loaded successfully with providers: {self.ort_session.get_providers()}")
            except Exception as e:
                print(f"Error loading ONNX model: {e}")
                raise

        # Setup image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def transform_image(self, image: Image.Image) -> np.ndarray:
        """Transform input image to numpy array for ONNX inference."""
        image = image.convert("RGB")
        tensor_image = self.transform(image).unsqueeze(0).numpy()
        return tensor_image

    def get_angles(self, numpy_image: np.ndarray) -> dict:
        """Get orientation probabilities from ONNX model."""
        input_name = self.ort_session.get_inputs()[0].name
        ort_inputs = {input_name: numpy_image}

        try:
            ort_outs = self.ort_session.run(None, ort_inputs)
            logits = ort_outs[0]
            probabilities = softmax(logits)[0]
            angles = {str(angle): float(score) for angle, score in zip(self.angles, probabilities)}
            return angles
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

    def predict(
            self,
            image: CogPath = Input(description="Image to detect orientation"),
            return_probabilities: bool = Input(description="Return all probabilities instead of just the best angle",
                                               default=False),
            use_rotation_averaging: bool = Input(description="Use rotation averaging to improve accuracy",
                                                 default=False)
    ) -> dict:
        """Predict image orientation."""
        print(f"Processing image: {image}")

        try:
            # Open and preprocess image
            img = Image.open(image).convert("RGB")
            numpy_image = self.transform_image(img)

            # Use rotation averaging if requested
            if use_rotation_averaging:
                accumulated = {a: [] for a in self.angles}
                base_pred = None

                for rotation in range(4):  # 0, 1, 2, 3 (number of 90° rotations)
                    angles = self.get_angles(numpy_image)

                    # Get the best angle for this rotation
                    best_angle = max(angles, key=lambda k: angles[k])
                    best_angle = int(best_angle)

                    if base_pred is None:
                        base_pred = best_angle

                    # Adjust each predicted angle to the absolute orientation
                    for pred_angle_str, score in angles.items():
                        pred_angle = int(pred_angle_str)
                        absolute_angle = (pred_angle - (rotation * 90)) % 360
                        accumulated[absolute_angle].append(score)

                    # Rotate the image by 90 degrees for next iteration
                    numpy_image = np.rot90(numpy_image, k=1, axes=(2, 3))

                # Average the scores for each absolute orientation
                result = {str(angle): float(sum(scores) / len(scores))
                          for angle, scores in accumulated.items()}
                best_angle = max(result, key=lambda k: result[k])

            else:
                # Standard single-pass prediction
                result = self.get_angles(numpy_image)
                best_angle = max(result, key=lambda k: result[k])

            # Return the result
            if return_probabilities:
                return {
                    "best_angle": int(best_angle),
                    "probabilities": result
                }
            else:
                return {"best_angle": int(best_angle)}

        except Exception as e:
            print(f"Error in prediction: {e}")
            return {"error": str(e)}