import re
import Levenshtein as Lv
from sentence_transformers import SentenceTransformer, util

# E0 patch: build the embedder lazily. The original loaded all-mpnet-base-v2 at
# import time, which aborts an offline E0 run (HF_HUB_OFFLINE=1) if the weights
# are not cached, even when --with_correction=false never uses it.
EMBED_MODEL = None


def _get_embed_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return EMBED_MODEL


_num_pat = re.compile(r"[-+]?\d+\.?\d*")

def _extract_nums(text):
    return [float(x) for x in _num_pat.findall(text)]

def mismatch_nums(stu: str, tch: str, tol: float = 0.10) -> bool:
    s_nums, t_nums = _extract_nums(stu), _extract_nums(tch)
    for n in t_nums:
        if not any(abs(n - s) / max(abs(n), 1e-3) <= tol for s in s_nums):
            return True
    return False

def wrong_formula(stu: str, tch: str) -> bool:
    patt = r"[=/*^]"
    s_form, t_form = re.search(patt + r".+", stu), re.search(patt + r".+", tch)
    if t_form and not s_form:                       
        return True
    if t_form and s_form:
        if Lv.ratio(s_form.group(0), t_form.group(0)) < 0.6:
            return True                          
    return False

def is_bad_answer(student: str, teacher: str, sim_th: float = 0.6, length_th: int = 25, fuzzy_threshold: float = 0.05) -> bool:


    if len(student.strip()) < length_th:
        return True

    model = _get_embed_model()
    emb_s = model.encode(student, convert_to_tensor=True)
    emb_t = model.encode(teacher, convert_to_tensor=True)
    sim = util.cos_sim(emb_s, emb_t).item()

    if abs(sim - sim_th) < fuzzy_threshold:
        return False

    bad = (sim < sim_th) or mismatch_nums(student, teacher) or wrong_formula(student, teacher)
    return bad


