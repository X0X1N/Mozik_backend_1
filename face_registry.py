import os, json, glob
import numpy as np
from typing import Dict, List, Tuple, Optional

class FaceRegistry:
    def __init__(self, db_path="face_db", sim_threshold=0.40, unknown_policy="neutral"):
        self.root = db_path
        self.threshold = float(sim_threshold)  # 코사인 거리 임계값(모델에 따라 조정)
        self.unknown_policy = unknown_policy
        os.makedirs(self.root, exist_ok=True)
        self.policy_path = os.path.join(self.root, "policy.json")
        self.policy = {"default": "neutral", "people": {}}  # neutral / exclude / force
        if os.path.exists(self.policy_path):
            try: self.policy = json.load(open(self.policy_path, "r", encoding="utf-8"))
            except: pass
        self._cache = {}  # name -> [embeddings]

    def save_policy(self):
        json.dump(self.policy, open(self.policy_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def add(self, name: str, emb: np.ndarray):
        p = os.path.join(self.root, name)
        os.makedirs(p, exist_ok=True)
        np.save(os.path.join(p, f"{len(glob.glob(os.path.join(p,'*.npy'))):04d}.npy"), emb.astype(np.float32))
        self._cache.pop(name, None)

    def set_policy(self, name: str, mode: str):
        # mode in {"neutral","exclude","force"}
        self.policy["people"][name] = mode
        self.save_policy()

    def get_policy(self, name: str) -> str:
        return self.policy["people"].get(name, self.policy.get("default","neutral"))

    def _load_embs(self, name: str) -> List[np.ndarray]:
        if name in self._cache: return self._cache[name]
        vecs = []
        for f in glob.glob(os.path.join(self.root, name, "*.npy")):
            try: vecs.append(np.load(f))
            except: pass
        self._cache[name] = vecs
        return vecs

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a) + 1e-9
        nb = np.linalg.norm(b) + 1e-9
        return 1.0 - float(np.dot(a, b) / (na*nb))  # 코사인 “거리”

    def match(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        best_name, best_d = None, 1e9
        for name in os.listdir(self.root):
            if name.startswith(".") or name == "policy.json": continue
            for ref in self._load_embs(name):
                d = self._cosine(emb, ref)
                if d < best_d: best_name, best_d = name, d
        if best_d <= self.threshold:
            return best_name, best_d
        return None, best_d
