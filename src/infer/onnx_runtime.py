from __future__ import annotations
from typing import Tuple
import numpy as np
import onnxruntime as ort


class ONNXSegRunner:
    def __init__(self, onnx_path: str, use_gpu: bool = True) -> None:
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            # needs onnxruntime-gpu
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def run(self, chw: np.ndarray) -> np.ndarray:
        """
        chw: float32, shape (1,3,H,W), normalized
        return logits: float32, (1,C,H,W)
        """
        out = self.sess.run([self.out_name], {self.in_name: chw})[0]
        return out
