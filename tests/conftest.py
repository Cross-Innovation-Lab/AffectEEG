"""Keep diagnostic tests bounded on shared workstations."""
import torch


def pytest_sessionstart(session):
    torch.set_num_threads(2)
