import torch

from badminton_decision.models.choice_head import ChoiceHead


def test_choice_head_variable_number_options():
    head = ChoiceHead(hidden_dim=32, head_dim=16, num_layers=2, num_task_types=4)
    choices = torch.randn(1, 5, 32)
    answer = torch.randn(1, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.bool)
    task = torch.tensor([0])
    logits = head(choices, answer, mask, task)
    assert logits.shape == (1, 5)


def test_choice_head_permutation_equivariance():
    torch.manual_seed(7)
    head = ChoiceHead(hidden_dim=32, head_dim=16, num_layers=2, num_task_types=4).eval()
    choices = torch.randn(1, 4, 32)
    answer = torch.randn(1, 32)
    mask = torch.ones(1, 4, dtype=torch.bool)
    task = torch.tensor([1])
    perm = torch.tensor([2, 0, 3, 1])
    a = head(choices, answer, mask, task)
    b = head(choices[:, perm], answer, mask, task)
    assert torch.allclose(b, a[:, perm], atol=1e-6)
