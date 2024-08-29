from .dmaq_qatten_learner import DMAQ_qattenLearner
from .nq_learner import NQLearner
from .nq_learner_data_augmentation import NQLearnerDataAugmentation
from .wm_learner import WMLearner
from .wm_v1_learner import WMV1Learner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["q_learner_data_augmentation"] = NQLearnerDataAugmentation
REGISTRY["wm_learner"] = WMLearner
REGISTRY["wm_v1_learner"] = WMV1Learner
