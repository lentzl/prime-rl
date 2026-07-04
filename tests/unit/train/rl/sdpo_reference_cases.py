"""Shared SDPO reference cases for trainer-side unit tests."""

import torch

REFERENCE_CASES = [
    {
        "name": "sampled_token_basic",
        "config": {
            "alpha": 1.0,
            "distillation_add_tail": False,
            "distillation_topk": None,
            "full_logit_distillation": False,
            "is_clip": None,
        },
        "expected_loss": -0.22750000655651093,
        "tensors": {
            "response_mask": [[True, True, False], [True, False, True]],
            "student_log_probs": [
                [-0.20000000298023224, -1.100000023841858, -0.699999988079071],
                [-0.4000000059604645, -0.8999999761581421, -1.2999999523162842],
            ],
            "teacher_log_probs": [
                [-0.10000000149011612, -1.399999976158142, -0.4000000059604645],
                [-0.6000000238418579, -0.5, -1.7000000476837158],
            ],
        },
    },
    {
        "name": "sampled_token_self_distillation_mask",
        "config": {
            "alpha": 1.0,
            "distillation_add_tail": False,
            "distillation_topk": None,
            "full_logit_distillation": False,
            "is_clip": None,
        },
        "expected_loss": -0.1549999713897705,
        "tensors": {
            "response_mask": [[True, True, False], [True, False, True]],
            "self_distillation_mask": [True, False],
            "student_log_probs": [
                [-0.20000000298023224, -1.100000023841858, -0.699999988079071],
                [-0.400000005960465, -0.8999999761581421, -1.2999999523162842],
            ],
            "teacher_log_probs": [
                [-0.10000000149011612, -1.399999976158142, -0.4000000059604645],
                [-0.6000000238418579, -0.5, -1.7000000476837158],
            ],
        },
    },
    {
        "name": "sampled_token_importance_clip",
        "config": {
            "alpha": 1.0,
            "distillation_add_tail": False,
            "distillation_topk": None,
            "full_logit_distillation": False,
            "is_clip": 1.2,
        },
        "expected_loss": -0.23946546018123627,
        "tensors": {
            "old_log_probs": [
                [-0.5, -1.0, -0.800000011920929],
                [-0.10000000149011612, -1.399999976158142, -1.899999976158142],
            ],
            "response_mask": [[True, True, False], [True, False, True]],
            "student_log_probs": [
                [-0.20000000298023224, -1.100000023841858, -0.699999988079071],
                [-0.4000000059604645, -0.8999999761581421, -1.2999999523162842],
            ],
            "teacher_log_probs": [
                [-0.10000000149011612, -1.399999976158142, -0.4000000059604645],
                [-0.6000000238418579, -0.5, -1.7000000476837158],
            ],
        },
    },
    {
        "name": "full_logit_forward_kl",
        "config": {
            "alpha": 0.0,
            "distillation_add_tail": False,
            "distillation_topk": None,
            "full_logit_distillation": True,
            "is_clip": None,
        },
        "expected_loss": 0.38995715975761414,
        "tensors": {
            "response_mask": [[True, True]],
            "student_all_log_probs": [
                [
                    [-1.435072898864746, -1.935072898864746, -0.6350728273391724, -2.435072898864746],
                    [-0.4234670400619507, -2.4234671592712402, -1.623466968536377, -2.823467254638672],
                ]
            ],
            "student_log_probs": [[-0.30000001192092896, -0.800000011920929]],
            "teacher_all_log_probs": [
                [
                    [-0.9254559278488159, -2.025455951690674, -1.2254558801651, -1.7254559993743896],
                    [-1.2512125968933105, -0.6512126326560974, -2.0512125492095947, -2.7512125968933105],
                ]
            ],
            "teacher_log_probs": [[-0.30000001192092896, -0.800000011920929]],
        },
    },
    {
        "name": "full_logit_reverse_kl",
        "config": {
            "alpha": 1.0,
            "distillation_add_tail": False,
            "distillation_topk": None,
            "full_logit_distillation": True,
            "is_clip": None,
        },
        "expected_loss": 0.3037019968032837,
        "tensors": {
            "response_mask": [[True, True]],
            "student_all_log_probs": [
                [
                    [-1.435072898864746, -1.935072898864746, -0.6350728273391724, -2.435072898864746],
                    [-0.4234670400619507, -2.4234671592712402, -1.623466968536377, -2.823467254638672],
                ]
            ],
            "student_log_probs": [[-0.30000001192092896, -0.800000011920929]],
            "teacher_all_log_probs": [
                [
                    [-0.9254559278488159, -2.025455951690674, -1.2254558801651, -1.7254559993743896],
                    [-1.2512125968933105, -0.6512126326560974, -2.0512125492095947, -2.7512125968933105],
                ]
            ],
            "teacher_log_probs": [[-0.30000001192092896, -0.800000011920929]],
        },
    },
    {
        "name": "full_logit_jsd_alpha_half",
        "config": {
            "alpha": 0.5,
            "distillation_add_tail": False,
            "distillation_topk": None,
            "full_logit_distillation": True,
            "is_clip": None,
        },
        "expected_loss": 0.08047773689031601,
        "tensors": {
            "response_mask": [[True, True]],
            "student_all_log_probs": [
                [
                    [-1.435072898864746, -1.935072898864746, -0.6350728273391724, -2.435072898864746],
                    [-0.4234670400619507, -2.4234671592712402, -1.623466968536377, -2.823467254638672],
                ]
            ],
            "student_log_probs": [[-0.30000001192092896, -0.800000011920929]],
            "teacher_all_log_probs": [
                [
                    [-0.9254559278488159, -2.025455951690674, -1.2254558801651, -1.7254559993743896],
                    [-1.2512125968933105, -0.6512126326560974, -2.0512125492095947, -2.7512125968933105],
                ]
            ],
            "teacher_log_probs": [[-0.30000001192092896, -0.800000011920929]],
        },
    },
    {
        "name": "topk_distillation_with_tail",
        "config": {
            "alpha": 1.0,
            "distillation_add_tail": True,
            "distillation_topk": 3,
            "full_logit_distillation": True,
            "is_clip": None,
        },
        "expected_loss": 0.04388948529958725,
        "tensors": {
            "response_mask": [[True, True]],
            "student_log_probs": [[-0.30000001192092896, -0.800000011920929]],
            "student_topk_log_probs": [
                [
                    [-0.7985077500343323, -1.3862943649291992, -2.3025851249694824],
                    [-1.0498220920562744, -1.6094379425048828, -1.8971199989318848],
                ]
            ],
            "teacher_log_probs": [[-0.30000001192092896, -0.800000011920929]],
            "teacher_topk_log_probs": [
                [
                    [-1.2039728164672852, -1.0498220920562744, -1.8971199989318848],
                    [-1.3862943649291992, -1.3862943649291992, -1.6094379425048828],
                ]
            ],
        },
    },
    {
        "name": "topk_tail_jsd_with_clip_and_token_rollout_is",
        "config": {
            "alpha": 0.5,
            "distillation_add_tail": True,
            "distillation_topk": 3,
            "full_logit_distillation": True,
            "is_clip": 2.0,
            "rollout_is": "token",
            "rollout_is_threshold": 2.0,
            "rollout_is_batch_normalize": False,
        },
        "expected_loss": 0.016971265897154808,
        "tensors": {
            "old_log_probs": [[-0.7000000476837158, -0.20000001788139343]],
            "response_mask": [[True, True]],
            "student_log_probs": [[-0.30000001192092896, -0.800000011920929]],
            "student_topk_log_probs": [
                [
                    [-0.7985077500343323, -1.3862943649291992, -2.3025851249694824],
                    [-1.0498220920562744, -1.6094379425048828, -1.8971199989318848],
                ]
            ],
            "teacher_log_probs": [[-0.30000001192092896, -0.800000011920929]],
            "teacher_topk_log_probs": [
                [
                    [-1.2039728164672852, -1.0498220920562744, -1.8971199989318848],
                    [-1.3862943649291992, -1.3862943649291992, -1.6094379425048828],
                ]
            ],
        },
    },
]


ROLLOUT_IS_WEIGHT_REFERENCE_CASES = [
    {
        "name": "token_rollout_is_batch_normalize_active_token_mean",
        "config": {
            "rollout_is": "token",
            "rollout_is_batch_normalize": True,
            "rollout_is_threshold": 2.0,
        },
        "expected_metrics": {
            "rollout_is_batch_norm_factor": 0.9375,
        },
        "expected_weights": [
            [2.133333444595337, 0.5333333611488342, 0.0],
            [1.0666667222976685, 0.2666666805744171, 0.0],
        ],
        "tensors": {
            "log_ratio": [
                [1.3862943649291992, -0.6931471824645996, 0.0],
                [0.0, -1.3862943649291992, 0.0],
            ],
            "response_mask": [
                [True, True, False],
                [True, True, False],
            ],
        },
    },
    {
        "name": "sequence_rollout_is_batch_normalize_active_sequence_mean",
        "config": {
            "rollout_is": "sequence",
            "rollout_is_batch_normalize": True,
            "rollout_is_threshold": 2.0,
        },
        "expected_metrics": {
            "rollout_is_batch_norm_factor": 1.25,
        },
        "expected_weights": [
            [1.600000023841858, 1.600000023841858, 0.0],
            [0.4000000059604645, 0.0, 0.0],
        ],
        "tensors": {
            "log_ratio": [
                [0.6931471824645996, 0.0, 0.0],
                [-0.6931471824645996, 0.0, 0.0],
            ],
            "response_mask": [
                [True, True, False],
                [True, False, False],
            ],
        },
    },
]


def _leaf(value):
    while isinstance(value, list):
        value = value[0]
    return value


def tensor_from_case_value(value):
    dtype = torch.bool if isinstance(_leaf(value), bool) else torch.float32
    return torch.tensor(value, dtype=dtype)
